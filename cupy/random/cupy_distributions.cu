#include <stdio.h>

// Add a state of the generator function
// that can be used to abstract both curand
// and custom generators

class rk_state {
public:
    __device__ virtual double rk_double() {
        return  0.0;
    }
};


__device__ double rk_standard_exponential(rk_state *state) {
    /* We use -log(1-U) since U is [0, 1) */
    return -log(1.0 - state->rk_double());
}


__global__ void generate_function() {
    return;
}
void standard_exponential() {
    generate_function<<<1,1>>>();
}
