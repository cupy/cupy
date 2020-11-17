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
// There are several errors when trying to do this a full template
// THIS CAN BE A PYTHON TEMPLATE
struct exponential_functor {
    template<typename... Args>
    __device__ double operator () (Args&&... args) {
        return rk_standard_exponential(std::forward<Args>(args)...);
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
