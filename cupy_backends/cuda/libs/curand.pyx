# distutils: language = c++

"""Thin wrapper of cuRAND."""
cimport cython  # NOQA

from cupy_backends.cuda.api cimport driver
from cupy_backends.cuda cimport stream as stream_module

###############################################################################
# Extern
###############################################################################

cdef extern from '../../cupy_rand.h' nogil:
    # Generator
    int curandCreateGenerator(Generator* generator, int rng_type)
    int curandDestroyGenerator(Generator generator)
    int curandGetVersion(int* version)

    # Stream
    int curandSetStream(Generator generator, driver.Stream stream)
    int curandSetPseudoRandomGeneratorSeed(
        Generator generator, unsigned long long seed)
    int curandSetGeneratorOffset(
        Generator generator, unsigned long long offset)
    int curandSetGeneratorOrdering(Generator generator, Ordering order)

    # Generation functions
    int curandGenerate(
        Generator generator, unsigned int* outputPtr, size_t num)
    int curandGenerateLongLong(
        Generator generator, unsigned long long* outputPtr, size_t num)
    int curandGenerateUniform(
        Generator generator, float* outputPtr, size_t num)
    int curandGenerateUniformDouble(
        Generator generator, double* outputPtr, size_t num)
    int curandGenerateNormal(
        Generator generator, float* outputPtr, size_t num,
        float mean, float stddev)
    int curandGenerateNormalDouble(
        Generator generator, double* outputPtr, size_t n,
        double mean, double stddev)
    int curandGenerateLogNormal(
        Generator generator, float* outputPtr, size_t n,
        float mean, float stddev)
    int curandGenerateLogNormalDouble(
        Generator generator, double* outputPtr, size_t n,
        double mean, double stddev)
    int curandGeneratePoisson(
        Generator generator, unsigned int* outputPtr, size_t n, double lam)


###############################################################################
# Error handling
###############################################################################

STATUS = {
    0: 'CURAND_STATUS_SUCCESS',
    100: 'CURAND_STATUS_VERSION_MISMATCH',
    101: 'CURAND_STATUS_NOT_INITIALIZED',
    102: 'CURAND_STATUS_ALLOCATION_FAILED',
    103: 'CURAND_STATUS_TYPE_ERROR',
    104: 'CURAND_STATUS_OUT_OF_RANGE',
    105: 'CURAND_STATUS_LENGTH_NOT_MULTIPLE',
    106: 'CURAND_STATUS_DOUBLE_PRECISION_REQUIRED',
    201: 'CURAND_STATUS_LAUNCH_FAILURE',
    202: 'CURAND_STATUS_PREEXISTING_FAILURE',
    203: 'CURAND_STATUS_INITIALIZATION_FAILED',
    204: 'CURAND_STATUS_ARCH_MISMATCH',
    999: 'CURAND_STATUS_INTERNAL_ERROR',
}


class CURANDError(RuntimeError):

    def __init__(self, status):
        self.status = status
        super(CURANDError, self).__init__(STATUS[status])

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CURANDError(status)


###############################################################################
# Generator
###############################################################################

cpdef size_t createGenerator(int rng_type) except? 0:
    cdef Generator generator
    with nogil:
        status = curandCreateGenerator(&generator, <RngType>rng_type)
    check_status(status)
    return <size_t>generator


cpdef destroyGenerator(size_t generator):
    status = curandDestroyGenerator(<Generator>generator)
    check_status(status)


cpdef int getVersion() except? -1:
    cdef int version
    status = curandGetVersion(&version)
    check_status(status)
    return version


cpdef setStream(size_t generator, size_t stream):
    status = curandSetStream(<Generator>generator, <driver.Stream>stream)
    check_status(status)


cdef _setStream(size_t generator):
    """Set current stream"""
    setStream(generator, stream_module.get_current_stream_ptr())


cpdef setPseudoRandomGeneratorSeed(size_t generator, unsigned long long seed):
    status = curandSetPseudoRandomGeneratorSeed(<Generator>generator, seed)
    check_status(status)


cpdef setGeneratorOffset(size_t generator, unsigned long long offset):
    status = curandSetGeneratorOffset(<Generator>generator, offset)
    check_status(status)


cpdef setGeneratorOrdering(size_t generator, int order):
    status = curandSetGeneratorOrdering(<Generator>generator, <Ordering>order)
    check_status(status)


###############################################################################
# Generation functions
###############################################################################

cpdef generate(size_t generator, size_t outputPtr, size_t num):
    _setStream(generator)
    status = curandGenerate(
        <Generator>generator, <unsigned int*>outputPtr, num)
    check_status(status)


cpdef generateLongLong(size_t generator, size_t outputPtr, size_t num):
    _setStream(generator)
    status = curandGenerateLongLong(
        <Generator>generator, <unsigned long long*>outputPtr, num)
    check_status(status)


cpdef generateUniform(size_t generator, size_t outputPtr, size_t num):
    _setStream(generator)
    status = curandGenerateUniform(
        <Generator>generator, <float*>outputPtr, num)
    check_status(status)


cpdef generateUniformDouble(size_t generator, size_t outputPtr, size_t num):
    _setStream(generator)
    status = curandGenerateUniformDouble(
        <Generator>generator, <double*>outputPtr, num)
    check_status(status)


cpdef generateNormal(size_t generator, size_t outputPtr, size_t n,
                     float mean, float stddev):
    if n % 2 == 1:
        msg = ('curandGenerateNormal can only generate even number of '
               'random variables simultaneously. See issue #390 for detail.')
        raise ValueError(msg)
    _setStream(generator)
    status = curandGenerateNormal(
        <Generator>generator, <float*>outputPtr, n, mean, stddev)
    check_status(status)


cpdef generateNormalDouble(size_t generator, size_t outputPtr, size_t n,
                           float mean, float stddev):
    if n % 2 == 1:
        msg = ('curandGenerateNormalDouble can only generate even number of '
               'random variables simultaneously. See issue #390 for detail.')
        raise ValueError(msg)
    _setStream(generator)
    status = curandGenerateNormalDouble(
        <Generator>generator, <double*>outputPtr, n, mean, stddev)
    check_status(status)


def generateLogNormal(size_t generator, size_t outputPtr, size_t n,
                      float mean, float stddev):
    if n % 2 == 1:
        msg = ('curandGenerateLogNormal can only generate even number of '
               'random variables simultaneously. See issue #390 for detail.')
        raise ValueError(msg)
    _setStream(generator)
    status = curandGenerateLogNormal(
        <Generator>generator, <float*>outputPtr, n, mean, stddev)
    check_status(status)


cpdef generateLogNormalDouble(size_t generator, size_t outputPtr, size_t n,
                              float mean, float stddev):
    if n % 2 == 1:
        msg = ('curandGenerateLogNormalDouble can only generate even number '
               'of random variables simultaneously. See issue #390 for '
               'detail.')
        raise ValueError(msg)
    _setStream(generator)
    status = curandGenerateLogNormalDouble(
        <Generator>generator, <double*>outputPtr, n, mean, stddev)
    check_status(status)


cpdef generatePoisson(size_t generator, size_t outputPtr, size_t n,
                      double lam):
    _setStream(generator)
    status = curandGeneratePoisson(
        <Generator>generator, <unsigned int*>outputPtr, n, lam)
    check_status(status)
