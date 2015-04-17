"""Default cuRAND generator"""

import os
import numpy
import pycuda.curandom as curandom
from pycuda import gpuarray

_seed = os.environ.get('CHAIN_SEED')

def _seed_getter(N):
    if _seed is None:
        return curandom.seed_getter_uniform(N)
    return gpuarray.empty((N,), dtype=numpy.int32).fill(_seed)

_generator = None
def get_generator():
    global _generator
    if _generator is None:
        _generator = curandom.XORWOWRandomNumberGenerator(seed_getter=_seed_getter)
    return _generator
