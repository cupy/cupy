#include <curand_kernel.h>
#include "cupy_distributions.h"
#include <utility>

// Add a state of the generator function
// that can be used to abstract both curand
// and custom generators

struct curand_xor_state: rk_state {
    // Valid for  XORWOW and MRG32k3a
    curandState_t state;
    __device__ virtual void init_state(int id, intptr_t param) {
        // TOOD(ecastill) enable reuse
        curand_init(static_cast<uint64_t>(param) + id, 0, 0, &state);
    }
    __device__ virtual uint32_t rk_int() {
        return  curand(&state);
    }
    __device__ virtual double rk_double() {
        return  curand_uniform(&state);
    }
    __device__ virtual double rk_normal() {
        return  curand_normal(&state);
    }
};


__device__ double rk_standard_exponential(rk_state* state) {
    /* We use -log(1-U) since U is [0, 1) */
    return -log(1.0 - state->rk_double());
}

__device__ double rk_standard_gamma(rk_state* state, double shape) {
    double b, c;
    double U, V, X, Y;
    if (shape == 1.0) {
        return rk_standard_exponential(state);
    } else if (shape < 1.0) {
        for (;;) {
            U = state->rk_double();
            V = rk_standard_exponential(state);
            if (U <= 1.0 - shape) {
                X = pow(U, 1./shape);
                if (X <= V) {
                    return X;
                }
            } else {
                Y = -log((1-U)/shape);
                X = pow(1.0 - shape + shape*Y, 1./shape);
                if (X <= (V + Y)) {
                    return X;
                }
            }
        }
    } else {
        b = shape - 1./3.;
        c = 1./sqrt(9*b);
        for (;;) {
            do {
                X = state->rk_normal();
                V = 1.0 + c*X;
            } while (V <= 0.0);
            V = V*V*V;
            U = state->rk_double();
            if (U < 1.0 - 0.0331*(X*X)*(X*X)) return (b*V);
            if (log(U) < 0.5*X*X + b*(1. - V + log(V))) return (b*V);
        }
    }
}

__device__ double rk_beta(rk_state* state, double a, double b) {
    double Ga, Gb;
    if ((a <= 1.0) && (b <= 1.0)) {
        double U, V, X, Y;
        /* Use Johnk's algorithm */
        while (1) {
            U = state->rk_double();
            V = state->rk_double();
            X = pow(U, 1.0/a);
            Y = pow(V, 1.0/b);
            if ((X + Y) <= 1.0) {
                if (X +Y > 0) {
                    return X / (X + Y);
                } else {
                    double logX = log(U) / a;
                    double logY = log(V) / b;
                    double logM = logX > logY ? logX : logY;
                    logX -= logM;
                    logY -= logM;
                    return exp(logX - log(exp(logX) + exp(logY)));
                }
            }
        }
    } else {
        Ga = rk_standard_gamma(state, a);
        Gb = rk_standard_gamma(state, b);
        return Ga/(Ga + Gb);
    }
}


__device__ uint32_t rk_interval_32(rk_state* state, int mx, int mask) {
    uint32_t sampled = state->rk_int() & mask;
    while(sampled > mx)  {
        sampled = state->rk_int() & mask;
    }
    return sampled;
}

__device__ uint64_t rk_interval_64(rk_state* state, uint64_t  mx, uint64_t mask) {
    uint32_t hi= state->rk_int();
    uint32_t lo= state->rk_int();
    uint64_t sampled = (static_cast<uint64_t>(hi) << 32 | lo)  & mask;
    while(sampled > mx)  {
        hi= state->rk_int();
        lo= state->rk_int();
        sampled = (static_cast<uint64_t>(hi) << 32 | lo) & mask;
    }
    return sampled;
}


struct interval_32_functor {
    template<typename... Args>
    __device__ uint32_t operator () (Args&&... args) {
        return rk_interval_32(args...);
    }
};

struct interval_64_functor {
    template<typename... Args>
    __device__ uint64_t operator () (Args&&... args) {
        return rk_interval_64(args...);
    }
};

struct beta_functor {
    template<typename... Args>
    __device__ double operator () (Args&&... args) {
        return rk_beta(args...);
    }
};

// There are several errors when trying to do this a full template
struct exponential_functor {
    template<typename... Args>
    __device__ double operator () (Args&&... args) {
        return rk_standard_exponential(args...);
    }
};

template<typename F, typename T, typename R, typename... Args>
__global__ void execute_dist(intptr_t param, void* out, ssize_t size, Args... args) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size) {
        T random;
        F func;
        random.init_state(id, param);
        ((R*) out)[id] = func(&random, args...);
    }
    return;
}

void interval_32(intptr_t param, int mx, int mask, void* out, ssize_t size) {
    int tpb = 256;
    int bpg =  (size + tpb - 1) / tpb;
    execute_dist<interval_32_functor, curand_xor_state, int32_t><<<bpg, tpb>>>(param, out, size, mx, mask);
}

void interval_64(intptr_t param, uint64_t mx, uint64_t mask, void* out, ssize_t size) {
    int tpb = 256;
    int bpg =  (size + tpb - 1) / tpb;
    execute_dist<interval_64_functor, curand_xor_state, int64_t><<<bpg, tpb>>>(param, out, size, mx, mask);
}

void beta(intptr_t  param, double a, double b, void* out, ssize_t size) {
    int tpb = 256;
    int bpg =  (size + tpb - 1) / tpb;
    execute_dist<beta_functor, curand_xor_state, double><<<bpg, tpb>>>(param, out, size, a, b);
}

void standard_exponential(intptr_t  param, void* out, ssize_t size) {
    int tpb = 256;
    int bpg =  (size + tpb - 1) / tpb;
    execute_dist<exponential_functor, curand_xor_state, double><<<bpg, tpb>>>(param, out, size);
}
