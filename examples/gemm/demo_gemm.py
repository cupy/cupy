import cupy
import numpy as np
import sys
import os.path as osp                  
import cubnn
from cubnn.kernel_utils import load_kernel, read_code
import math

import functools
from gemm_configs import configs


types_dict = {'unsigned int': np.uint32}


def empty_f(shape, dtype):
    x = cupy.empty(shape[::-1], dtype=dtype)
    return x.T


def bitgemm(A, B, C=None):
    m, k = A.shape
    k, n = B.shape
    out_t = 'unsigned int'
    assert A.dtype == np.uint32
    assert B.dtype == np.uint32

    if C is None:
        C = cupy.empty((m, n), dtype=types_dict[out_t], order='F')

    config = configs['16d_64x64x4']
    #config = configs['nn_1']
    config.update({'THR_M': config['BLK_M'] / config['DIM_X'],
                   'THR_N': config['BLK_N'] / config['DIM_Y'],
                   'USE_BINARY': 1})
    bitgemm_func = bitgemm_core(A, B, C, config)
    bitgemm_func()
    return C


def bitgemm_core(A, B, C, config={}):
    m, k = A.shape
    k, n = B.shape

    assert C.flags.f_contiguous
    print config

    # this limitation should be removed
    #assert m % config['BLK_M'] == 0
    #assert n % config['BLK_N'] == 0
    #assert k % config['BLK_K'] == 0

    # you need valid distribution of DIM_X * DIM_Y threads to blocks for
    # A and B
    #assert (config['DIM_X'] * config['DIM_Y']
    #        == config['DIM_XA'] * config['DIM_YA']
    #        == config['DIM_XB'] * config['DIM_YB']) 

    A = cupy.asfortranarray(A)
    B = cupy.asfortranarray(B)

    code = read_code(cubnn.bitgemm_file, params=config)
    kern = load_kernel('bitgemm', code)

    grid = (math.ceil(m / float(config['BLK_M'])),
            math.ceil(n / float(config['BLK_N'])), 1)
    block = (config['DIM_X'], config['DIM_Y'], 1)
    args = (m, n, k,
            A, m,
            B, k,
            C, m)
    shared_mem = config['BLK_K'] * (config['BLK_M'] + 1) * 4 +\
        config['BLK_N'] * (config['BLK_K'] + 1) * 4
    func = functools.partial(
        kern, grid, block, args=args, shared_mem=shared_mem)
    return func
