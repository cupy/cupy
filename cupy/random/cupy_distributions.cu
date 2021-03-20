#include <stdio.h>
#include <stdexcept>
#include <utility>
#include <iostream>

#include <curand_kernel.h>

#include "cupy_distributions.cuh"


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

template<typename CURAND_TYPE>
struct curand_pseudo_state: rk_state {
    // Valid for  XORWOW and MRG32k3a
    CURAND_TYPE* _state;
    int _id;

    __device__ curand_pseudo_state(int id, intptr_t state) {
        _state = reinterpret_cast<CURAND_TYPE*>(state) + id;
        _id = id;
    }
    __device__ virtual uint32_t rk_int() {
        return curand(_state);
    }
    __device__ virtual double rk_double() {
        // Curand returns (0, 1] while the functions
        // below rely on [0, 1)
        double r = curand_uniform(_state);
        if (r >= 1.0) { 
           r = 0.0;
        }
        return r;
    }
    __device__ virtual double rk_normal() {
        return curand_normal(_state);
    }
};


// This design is the same as the dtypes one
template <typename F, typename... Ts>
void generator_dispatcher(int generator_id, F f, Ts&&... args) {
   switch(generator_id) {
       case CURAND_XOR_WOW: return f.template operator()<curand_pseudo_state<curandState>>(std::forward<Ts>(args)...);
       case CURAND_MRG32k3a: return f.template operator()<curand_pseudo_state<curandStateMRG32k3a>>(std::forward<Ts>(args)...);
       case CURAND_PHILOX_4x32_10: return f.template operator()<curand_pseudo_state<curandStatePhilox4_32_10_t>>(std::forward<Ts>(args)...);
       default: throw std::runtime_error("Unknown random generator");
   }
}


template<typename T>
__global__ void init_curand(intptr_t state, uint64_t seed, ssize_t size) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    T curand_state(id, state);
    if (id < size) {
        curand_init(seed, id, 0, curand_state._state);    
    }
}

struct initialize_launcher {
    initialize_launcher(ssize_t size, cudaStream_t stream) : _size(size), _stream(stream) {
    }
    template<typename T, typename... Args>
    void operator()(Args&&... args) { 
        int tpb = 256;
        int bpg =  (_size + tpb - 1) / tpb;
        init_curand<T><<<bpg, tpb, 0, _stream>>>(std::forward<Args>(args)...);
    }
    ssize_t _size;
    cudaStream_t _stream;
};

void init_curand_generator(int generator, intptr_t state_ptr, uint64_t seed, ssize_t size, intptr_t stream) {
    // state_ptr is a device ptr
    initialize_launcher launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state_ptr, seed, size);
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

__device__ long rk_poisson_mult(rk_state *state, double lam) {
    long X;
    double prod, U, enlam;
    enlam = exp(-lam);
    X = 0;
    prod = 1.0;
    while (1) {
        U = state->rk_double();
        prod *= U;
        if (prod > enlam) {
            X += 1;
        } else {
            return X;
        }
    }
}

__device__ long rk_poisson_ptrs(rk_state *state, double lam) {
    long k;
    double U, V, slam, loglam, a, b, invalpha, vr, us;
    slam = sqrt(lam);
    loglam = log(lam);
    b = 0.931 + 2.53*slam;
    a = -0.059 + 0.02483*b;
    invalpha = 1.1239 + 1.1328/(b-3.4);
    vr = 0.9277 - 3.6224/(b-2);
    while (1) {
        U = state->rk_double(); - 0.5;
        V = state->rk_double();
        us = 0.5 - fabs(U);
        k = (long)floor((2*a/us + b)*U + lam + 0.43);
        if ((us >= 0.07) && (V <= vr)) {
            return k;
        }
        if ((k < 0) ||
            ((us < 0.013) && (V > us))) {
            continue;
        }
        if ((log(V) + log(invalpha) - log(a/(us*us)+b)) <=
            (-lam + k*loglam - loggam(k+1))) {
            return k;
        }
    }
}

__device__ long rk_poisson(rk_state *state, double lam) {
    if (lam >= 10) {
        return rk_poisson_ptrs(state, lam);
    } else if (lam == 0) {
        return 0;
    } else {
        return rk_poisson_mult(state, lam);
    }
}

__device__ uint32_t rk_raw(rk_state* state) {
    return state->rk_int();
}

__device__ uint32_t rk_interval_32(rk_state* state, uint32_t mx, uint32_t mask) {
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


struct raw_functor {
    template<typename... Args>
    __device__ uint32_t operator () (Args&&... args) {
        return rk_raw(args...);
    }
};


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

struct poisson_functor {
    template<typename... Args>
    __device__ double operator () (Args&&... args) {
        return rk_poisson(args...);
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

template <typename F, typename R>
struct kernel_launcher {
    kernel_launcher(ssize_t size, cudaStream_t stream) : _size(size), _stream(stream) {
    }
    template<typename T, typename... Args>
    void operator()(Args&&... args) { 
        int tpb = 256;
        int bpg =  (_size + tpb - 1) / tpb;
        execute_dist<F, T, R><<<bpg, tpb, 0, _stream>>>(std::forward<Args>(args)...);
    }
    ssize_t _size;
    cudaStream_t _stream;
};

//These functions will take the generator_id as a parameter
void raw(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {
    kernel_launcher<raw_functor, int32_t> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size);
}

//These functions will take the generator_id as a parameter
void interval_32(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, int32_t mx, int32_t mask) {
    kernel_launcher<interval_32_functor, int32_t> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size, static_cast<uint32_t>(mx), static_cast<uint32_t>(mask));
}

void interval_64(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, int64_t mx, int64_t mask) {
    kernel_launcher<interval_64_functor, int64_t> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size, static_cast<uint64_t>(mx), static_cast<uint64_t>(mask));
}

void beta(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, double a, double b) {
    kernel_launcher<beta_functor, double> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size, a, b);
}

void exponential(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {
    kernel_launcher<exponential_functor, double> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size);
}

void poisson(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, double lam) {
    kernel_launcher<poisson_functor, double> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size, lam);
}

