"""
Data transfer methods
"""

import cupy as cp
import numpy as np


def vram2ram(args, kwargs):
    """
    Transfers ndarrays in *args, **kwargs from GPU to CPU.

    Args:
        args (tuple): Arguments.
        kwargs (dict): Keyword arguments.

    Returns:
        cpu_args (tuple): Arguments in CPU.
        cpu_kwargs (dict): Keyword arguments in CPU.
    """
    cpu_args = []
    cpu_kwargs = {}

    for arg in args:
        if isinstance(arg, cp.ndarray):
            cpu_args.append(cp.asnumpy(arg))
        else:
            cpu_args.append(arg)

    for arg_name, arg in kwargs.items():
        if isinstance(arg, cp.ndarray):
            cpu_kwargs[arg_name] = cp.asnumpy(arg)
        else:
            cpu_kwargs[arg_name] = arg

    return tuple(cpu_args), cpu_kwargs


def ram2vram(res):
    """
    Transfers ndarrays in *args, **kwargs from CPU to GPU.

    Args:
        res (tuple, list, numpy-ndarray): Result by executing numpy_func.

    Returns:
        gpu_res (tuple, list, cupy-ndarray): Result transfered to GPU.
    """
    if isinstance(res, (list, tuple)):
        gpu_res = []
        for r in res:
            if isinstance(r, np.ndarray):
                gpu_res.append(cp.array(r))
            else:
                gpu_res.append(r)
    else:
        if isinstance(res, np.ndarray):
            return cp.array(res)
        return res
