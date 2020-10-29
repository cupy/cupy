#include <curand_kernel.h>
#include "cupy_distributions.cuh"
#include <utility>
#include <stdio.h>

// Add a state of the generator function
// that can be used to abstract both curand
// and custom generators


struct rk_state {

    __device__ virtual uint32_t rk_int() {
        return  0;
    }
    __device__ virtual double rk_double() {
        return  0.0;
    }
    __device__ virtual double rk_normal() {
        return  0.0;
    }
};


struct curand_xor_state: rk_state {
    // Valid for  XORWOW and MRG32k3a
    curandState* _state;
    int _id;

    __device__ curand_xor_state(int id, intptr_t state) {
        _state = reinterpret_cast<curandState*>(state) + id;
        _id = id;
    }
    __device__ virtual uint32_t rk_int() {
        return curand(_state);
    }
    __device__ virtual double rk_double() {
        return curand_uniform(_state);
    }
    __device__ virtual double rk_normal() {
        return curand_normal(_state);
    }
};


__global__ void init_xor(intptr_t state, uint64_t seed, ssize_t size) {
    curandState* state_ptr = reinterpret_cast<curandState*>(state);
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    if (id < size) {
        curand_init(seed, id, 0, &state_ptr[id]);    
    }
}

void init_xor_generator(intptr_t state_ptr, uint64_t seed, ssize_t size, intptr_t stream) {
    // state_ptr is a device ptr
    int tpb = 256;
    int bpg =  (size + tpb - 1) / tpb;
    cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
    init_xor<<<bpg, tpb, 0, stream_>>>(state_ptr, seed, size);
}

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
        return rk_interval_32(std::forward<Args>(args)...);
    }
};

struct interval_64_functor {
    template<typename... Args>
    __device__ uint64_t operator () (Args&&... args) {
        return rk_interval_64(std::forward<Args>(args)...);
    }
};

struct beta_functor {
    template<typename... Args>
    __device__ double operator () (Args&&... args) {
        return rk_beta(std::forward<Args>(args)...);
    }
};

// There are several errors when trying to do this a full template
struct exponential_functor {
    template<typename... Args>
    __device__ double operator () (Args&&... args) {
        return rk_standard_exponential(std::forward<Args>(args)...);
    }
};

template<typename F, typename T, typename R, typename... Args>
__global__ void execute_dist(intptr_t state, intptr_t out, ssize_t size, Args... args) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    R* out_ptr = reinterpret_cast<R*>(out);
    if (id < size) {
        T random(id, state);
        F func;
        out_ptr[id] = func(&random, std::forward<Args>(args)...);
    }
    return;
}

void interval_32(intptr_t state, intptr_t out, ssize_t size, intptr_t stream, int mx, int mask) {
    int tpb = 256;
    int bpg =  (size + tpb - 1) / tpb;
    cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
    execute_dist<interval_32_functor, curand_xor_state, int32_t><<<bpg, tpb, 0, stream_>>>(state, out, size, mx, mask);
}

void interval_64(intptr_t state, intptr_t out, ssize_t size, intptr_t stream, uint64_t mx, uint64_t mask) {
    int tpb = 256;
    int bpg =  (size + tpb - 1) / tpb;
    cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
    execute_dist<interval_64_functor, curand_xor_state, int64_t><<<bpg, tpb, 0, stream_>>>(state, out, size, mx, mask);
}

void beta(intptr_t state, intptr_t out, ssize_t size, intptr_t stream, double a, double b) {
    int tpb = 256;
    int bpg =  (size + tpb - 1) / tpb;
    cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
    execute_dist<beta_functor, curand_xor_state, double><<<bpg, tpb, 0, stream_>>>(state, out, size, a, b);
}

void standard_exponential(intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {
    int tpb = 256;
    int bpg =  (size + tpb - 1) / tpb;
    cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
    execute_dist<exponential_functor, curand_xor_state, double><<<bpg, tpb, 0, stream_>>>(state, out, size);
}
