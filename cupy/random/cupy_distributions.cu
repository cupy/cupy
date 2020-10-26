#include <curand_kernel.h>
#include "cupy_distributions.h"

// Add a state of the generator function
// that can be used to abstract both curand
// and custom generators

struct rk_state {
    __device__ virtual void init_state(int id, intptr_t param);
    __device__ virtual double rk_double();
    __device__ virtual double rk_normal();
};

struct curand_pseudorand_state: rk_state {
    // Valid for  XORWOW and MRG32k3a
    curandState_t state;
    __device__ virtual void init_state(int id, intptr_t param) {
        // TOOD(ecastill) enable reuse
        curand_init(static_cast<uint64_t>(param) + id, 0, 0, &state);
    }
    __device__ virtual double rk_double() {
        return  curand_uniform(&state);
    }
    __device__ virtual double rk_normal() {
        return  curand_normal(&state);
    }
};

struct curand_mtgp32_state: rk_state {
    // This is initialized from the host and given as a pointer
    // to the device
    curandStateMtgp32_t* state;
    __device__ virtual void init_state(int id, intptr_t param) {
        state = reinterpret_cast<curandStateMtgp32_t*>(param);
    }
    __device__ virtual double rk_double() {
        return  curand_uniform(state);
    }
    __device__ virtual double rk_normal() {
        return  curand_normal(state);
    }
};


__device__ double rk_standard_exponential(rk_state *state) {
    /* We use -log(1-U) since U is [0, 1) */
    return -log(1.0 - state->rk_double());
}

__device__ double rk_standard_gamma(rk_state *state, double shape) {
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


__device__ double rk_beta(rk_state *state, double a, double b) {
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



template<typename T>
__global__ void standard_exponential_kernel(intptr_t param, void* out, ssize_t size) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size) {
        T random;
        random.init_state(id, param);
        ((double*) out)[id] = rk_standard_exponential(&random);
    }
    return;
}



void standard_exponential(intptr_t handle, intptr_t  param, void* out, ssize_t size) {
    int tpb = 256;
    int bpg =  (size + tpb - 1) / tpb;
    standard_exponential_kernel<curand_pseudorand_state><<<bpg, tpb>>>(param, out, size);
}
