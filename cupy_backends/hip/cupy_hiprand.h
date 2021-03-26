#ifndef INCLUDE_GUARD_HIP_CUPY_HIPRAND_H
#define INCLUDE_GUARD_HIP_CUPY_HIPRAND_H

#include <hiprand/hiprand.h>
#include "cupy_hip_common.h"

extern "C" {

typedef enum {} curandOrdering_t;
typedef hiprandRngType curandRngType_t;
typedef hiprandStatus_t curandStatus_t;

typedef hiprandGenerator_t curandGenerator_t;

curandRngType_t convert_hiprandRngType(curandRngType_t t) {
    switch(static_cast<int>(t)) {
    case 100: return HIPRAND_RNG_PSEUDO_DEFAULT;
    case 101: return HIPRAND_RNG_PSEUDO_XORWOW;
    case 121: return HIPRAND_RNG_PSEUDO_MRG32K3A;
    case 141: return HIPRAND_RNG_PSEUDO_MTGP32;
    case 142: return HIPRAND_RNG_PSEUDO_MT19937;
    case 161: return HIPRAND_RNG_PSEUDO_PHILOX4_32_10;
    case 200: return HIPRAND_RNG_QUASI_DEFAULT;
    case 201: return HIPRAND_RNG_QUASI_SOBOL32;
    case 202: return HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32;
    case 203: return HIPRAND_RNG_QUASI_SOBOL64;
    case 204: return HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64;
    }
    return HIPRAND_RNG_TEST;
}

// curandGenerator_t
curandStatus_t curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type) {
    rng_type = convert_hiprandRngType(rng_type);
    return hiprandCreateGenerator(generator, rng_type);
}

curandStatus_t curandDestroyGenerator(curandGenerator_t generator) {
    return hiprandDestroyGenerator(generator);
}

curandStatus_t curandGetVersion(int *version) {
    return hiprandGetVersion(version);
}


// Stream
curandStatus_t curandSetStream(curandGenerator_t generator, cudaStream_t stream) {
    return hiprandSetStream(generator, stream);
}

curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed) {
    return hiprandSetPseudoRandomGeneratorSeed(generator, seed);
}

curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset) {
    return hiprandSetGeneratorOffset(generator, offset);
}

curandStatus_t curandSetGeneratorOrdering(...) {
    return HIPRAND_STATUS_NOT_IMPLEMENTED;
}


// Generation functions
curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int *output_data, size_t n) {
    return hiprandGenerate(generator, output_data, n);
}

curandStatus_t curandGenerateLongLong(...) {
    return HIPRAND_STATUS_NOT_IMPLEMENTED;
}

curandStatus_t curandGenerateUniform(curandGenerator_t generator, float *output_data, size_t n) {
    return hiprandGenerateUniform(generator, output_data, n);
}

curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator, double *output_data, size_t n) {
    return hiprandGenerateUniformDouble(generator, output_data, n);
}

curandStatus_t curandGenerateNormal(curandGenerator_t generator, float *output_data, size_t n, float mean, float stddev) {
    return hiprandGenerateNormal(generator, output_data, n, mean, stddev);
}

curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, double *output_data, size_t n, double mean, double stddev) {
    return hiprandGenerateNormalDouble(generator, output_data, n, mean, stddev);
}

curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, float *output_data, size_t n, float mean, float stddev) {
    return hiprandGenerateLogNormal(generator, output_data, n, mean, stddev);
}

curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator, double *output_data, size_t n, double mean, double stddev) {
    return hiprandGenerateLogNormalDouble(generator, output_data, n, mean, stddev);
}

curandStatus_t curandGeneratePoisson(curandGenerator_t generator, unsigned int *output_data, size_t n, double lambda) {
    return hiprandGeneratePoisson(generator, output_data, n, lambda);
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_HIPRAND_H
