"""Thin wrapper of cuRAND."""
import ctypes
import sys

from cupy.cuda import internal
from cupy.cuda import runtime

if 'win32' == sys.platform:
    _curand = internal.load_library(
        internal.get_windows_cuda_library_names('curand'))
else:
    _curand = internal.load_library('curand')

_I = ctypes.c_int
_U = ctypes.c_uint
_S = ctypes.c_size_t
_ULL = ctypes.c_ulonglong
_P = ctypes.c_void_p
_IP = ctypes.POINTER(_I)
_UP = ctypes.POINTER(_U)
_ULLP = ctypes.POINTER(_ULL)
_F = ctypes.c_float
_D = ctypes.c_double
_FP = ctypes.POINTER(_F)
_DP = ctypes.POINTER(_D)

CURAND_RNG_PSEUDO_DEFAULT = 100
CURAND_RNG_PSEUDO_XORWOW = 101
CURAND_RNG_PSEUDO_MRG32K3A = 121
CURAND_RNG_PSEUDO_MTGP32 = 141
CURAND_RNG_PSEUDO_MT19937 = 142
CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161
CURAND_RNG_QUASI_DEFAULT = 200
CURAND_RNG_QUASI_SOBOL32 = 201
CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202
CURAND_RNG_QUASI_SOBOL64 = 203
CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204

CURAND_ORDERING_PSEUDO_BEST = 100
CURAND_ORDERING_PSEUDO_DEFAULT = 101
CURAND_ORDERING_PSEUDO_SEEDED = 102
CURAND_ORDERING_QUASI_DEFAULT = 201

Generator = _P
Distribution = _DP

DiscreteDistribution = _P

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


def check_status(status):
    if status != 0:
        raise CURANDError(status)

###############################################################################
# Generator
###############################################################################

_curand.curandCreateGenerator.argtypes = (_P, _I)


def createGenerator(rng_type):
    generator = Generator()
    status = _curand.curandCreateGenerator(ctypes.byref(generator), rng_type)
    check_status(status)
    return generator


_curand.curandDestroyGenerator.argtypes = (Generator,)


def destroyGenerator(generator):
    status = _curand.curandDestroyGenerator(generator)
    check_status(status)


_curand.curandGetVersion.argtypes = (_IP,)


def getVersion():
    version = _I()
    status = _curand.curandGetVersion(ctypes.byref(version))
    check_status(status)
    return version


_curand.curandSetStream.argtypes = (Generator, runtime.Stream)


def setStream(generator, stream):
    status = _curand.curandSetStream(generator, stream)
    check_status(status)


_curand.curandSetPseudoRandomGeneratorSeed.argtypes = (Generator, _ULL)


def setPseudoRandomGeneratorSeed(generator, seed):
    status = _curand.curandSetPseudoRandomGeneratorSeed(generator, seed)
    check_status(status)


_curand.curandSetGeneratorOffset.argtypes = (Generator, _ULL)


def setGeneratorOffset(generator, offset):
    status = _curand.curandSetGeneratorOffset(generator, offset)
    check_status(status)


_curand.curandSetGeneratorOrdering.argtypes = (Generator, _I)


def setGeneratorOrdering(generator, order):
    status = _curand.curandSetGeneratorOrdering(generator, order)
    check_status(status)


###############################################################################
# Generation functions
###############################################################################

_curand.curandGenerate.argtypes = (Generator, _P, _S)


def generate(generator, outputPtr, num):
    status = _curand.curandGenerate(generator, outputPtr, num)
    check_status(status)


_curand.curandGenerateLongLong.argtypes = (Generator, _P, _S)


def generateLongLong(generator, outputPtr, num):
    status = _curand.curandGenerateLongLong(generator, outputPtr, num)
    check_status(status)


_curand.curandGenerateUniform.argtypes = (Generator, _P, _S)


def generateUniform(generator, outputPtr, num):
    status = _curand.curandGenerateUniform(generator, outputPtr, num)
    check_status(status)


_curand.curandGenerateUniformDouble.argtypes = (Generator, _P, _S)


def generateUniformDouble(generator, outputPtr, num):
    status = _curand.curandGenerateUniformDouble(generator, outputPtr, num)
    check_status(status)


_curand.curandGenerateNormal.argtypes = (Generator, _P, _S, _F, _F)


def generateNormal(generator, outputPtr, n, mean, stddev):
    if n % 2 == 1:
        msg = 'curandGenerateNormal can only generate even number of '\
              'random variables simultaneously. See issue #390 for detail.'
        raise ValueError(msg)
    status = _curand.curandGenerateNormal(generator, outputPtr, n, mean,
                                          stddev)
    check_status(status)


_curand.curandGenerateNormalDouble.argtypes = (Generator, _P, _S, _D, _D)


def generateNormalDouble(generator, outputPtr, n, mean, stddev):
    if n % 2 == 1:
        msg = 'curandGenerateNormalDouble can only generate even number of '\
              'random variables simultaneously. See issue #390 for detail.'
        raise ValueError(msg)
    status = _curand.curandGenerateNormalDouble(generator, outputPtr, n, mean,
                                                stddev)
    check_status(status)


_curand.curandGenerateLogNormal.argtypes = (Generator, _P, _S, _F, _F)


def generateLogNormal(generator, outputPtr, n, mean, stddev):
    if n % 2 == 1:
        msg = 'curandGenerateLogNormal can only generate even number of '\
              'random variables simultaneously. See issue #390 for detail.'
        raise ValueError(msg)
    status = _curand.curandGenerateLogNormal(generator, outputPtr, n,
                                             mean, stddev)
    check_status(status)


_curand.curandGenerateLogNormalDouble.argtypes = (Generator, _P, _S, _D, _D)


def generateLogNormalDouble(generator, outputPtr, n, mean, stddev):
    if n % 2 == 1:
        msg = 'curandGenerateLogNormalDouble can only generate even number of '\
              'random variables simultaneously. See issue #390 for detail.'
        raise ValueError(msg)
    status = _curand.curandGenerateLogNormalDouble(generator, outputPtr, n,
                                                   mean, stddev)
    check_status(status)


_curand.curandGeneratePoisson.argtypes = (Generator, _P, _S, _D)


def generatePoisson(generator, outputPtr, n, lam):
    status = _curand.curandGeneratePoisson(generator, outputPtr, n, lam)
    check_status(status)
