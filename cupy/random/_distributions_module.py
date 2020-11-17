import cupy

distributions_code = """
#include <curand_kernel.h>
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
        return curand_uniform(_state);
    }
    __device__ virtual double rk_normal() {
        return curand_normal(_state);
    }
};

// Use template specialization for custom ones
template<typename T>
__global__ void init_generator(intptr_t state, uint64_t seed, uint64_t size) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    T curand_state(id, state);
    if (id < size) {
        curand_init(seed, id, 0, curand_state._state);
    }
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

// There are several errors when trying to do this a full template
// THIS CAN BE A PYTHON TEMPLATE
struct exponential_functor {
    template<typename... Args>
    __device__ double operator () (Args&&... args) {
        return rk_standard_exponential(std::forward<Args>(args)...);
    }
};

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

template<typename F, typename T, typename R, typename... Args>
__device__ void execute_dist(intptr_t state, intptr_t out, uint64_t size, Args... args) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    R* out_ptr = reinterpret_cast<R*>(out);
    if (id < size) {
        T random(id, state);
        F func;
        out_ptr[id] = func(&random, std::forward<Args>(args)...);
    }
    return;
}

template<typename T>
__global__ void interval_32(intptr_t state, intptr_t out, uint64_t size, uint32_t mx, uint32_t mask) {
    execute_dist<interval_32_functor, T, int32_t>(state, out, size, mx, mask);
}

template<typename T>
__global__ void interval_64(intptr_t state, intptr_t out, uint64_t size, uint64_t mx, uint64_t mask) {
    execute_dist<interval_64_functor, T, int64_t>(state, out, size, mx, mask);
}

template<typename T>
__global__ void beta(intptr_t state, intptr_t out, uint64_t size, double a, double b) {
    execute_dist<beta_functor, T, double>(state, out, size, a, b);
}

// T is the generator type it is overriden by python when compiling
template<typename T>
__global__ void exponential(intptr_t state, intptr_t out, uint64_t size) {
    execute_dist<exponential_functor, T, double>(state, out, size);
}
"""


@cupy._util.memoize(for_each_device=True)
def _get_distributions_module(c_type_generator):
    code = distributions_code
    name_expressions = [f'init_generator<{c_type_generator}>',
                        f'beta<{c_type_generator}>',
                        f'interval_32<{c_type_generator}>',
                        f'interval_64<{c_type_generator}>',
                        f'exponential<{c_type_generator}>']
    module = cupy.RawModule(code=code, options=('--std=c++11',),
                            name_expressions=name_expressions, jitify=True)
    return module


@cupy._util.memoize(for_each_device=True)
def _get_distribution(generator, distribution):
    c_generator = generator._c_layer_generator()
    module = _get_distributions_module(generator._c_layer_generator())
    kernel = module.get_function(f'{distribution}<{c_generator}>')
    return kernel


@cupy._util.memoize(for_each_device=True)
def _initialize_generator(generator):
    c_generator = generator._c_layer_generator()
    module = _get_distributions_module(generator._c_layer_generator())
    kernel = module.get_function(f'init_generator<{c_generator}>')
    return kernel
