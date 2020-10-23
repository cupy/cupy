#include <curand_kernel.h>
#include "cupy_distributions.h"

// Add a state of the generator function
// that can be used to abstract both curand
// and custom generators

struct rk_state {
    __device__ virtual void init_state(int id, uint64_t seed);
    __device__ virtual double rk_double();
};

struct curand_basic_state: rk_state {
    curandState_t state;
    __device__ virtual void init_state(int id, uint64_t seed) {
        curand_init(seed + id, 0, 0, &state);
    }
    __device__ virtual double rk_double() {
        return  curand_uniform(&state);
    }
};

__device__ double rk_standard_exponential(rk_state *state) {
    /* We use -log(1-U) since U is [0, 1) */
    return -log(1.0 - state->rk_double());
}


template<typename T>
__global__ void call_function(int64_t seed, void* out, ssize_t size) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size) {
        T random;
        random.init_state(id, seed);
        ((double*) out)[id] = rk_standard_exponential(&random);
    }
    return;
}


void standard_exponential(intptr_t handle, uint64_t seed, void* out, ssize_t size){
    int tpb = 256;
    int bpg =  (size + tpb - 1) / tpb;
    // Use an actual C type
    call_function<curand_basic_state><<<bpg, tpb>>>(seed, out, size);
}
