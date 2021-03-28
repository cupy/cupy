import functools

import numpy

from cupy._core import core
from cupy._core import _fusion_trace
from cupy._core import _fusion_kernel
from cupy._core import _fusion_thread_local
from cupy._core._fusion_interface import _ScalarProxy  # NOQA
from cupy._core._fusion_interface import _ArrayProxy  # NOQA
from cupy._core._fusion_variable import _AbstractDim
from cupy._core._dtype cimport get_dtype


_thread_local = _fusion_thread_local.thread_local
_is_fusing = _fusion_thread_local.is_new_fusing
_call_ufunc = _fusion_thread_local.call_ufunc
_call_reduction = _fusion_thread_local.call_reduction


cdef tuple _fusion_argument_types = (
    core.ndarray, numpy.ndarray, numpy.generic,
    int, float, complex, bool, type(None))


def _get_fused_kernel(name, func, args):
    try:
        _thread_local.is_new_fusing = True
        trace_result = _fusion_trace.trace(func, args)
        kernel = _fusion_kernel.FusedKernel(name, trace_result)
    finally:
        _thread_local.is_new_fusing = False
    return kernel


class Fusion:
    """Function class.

    This class can be get by using `fuse` function

    Args:
        func (function): The function before fusing.
        name (str optional): The name of the function.
    """
    def __init__(self, func, name=None):
        self.func = func
        self.name = name or func.__name__
        self._cache = {}
        # TODO(asi1024): Support switch of optimization mode.

    def __repr__(self):
        return '<Fusion name={}>'.format(self.name)

    def clear_cache(self):
        self._cache = {}

    def __call__(self, *args, **kwargs):
        cdef int nargs = len(args)
        cdef int i

        if _is_fusing():
            # Inner function of composition of multiple fused functions.
            return self.func(*args)

        exec_cupy = False
        for i in range(nargs):
            if isinstance(args[i], core.ndarray):
                exec_cupy = True
                break
        if not exec_cupy:
            # No cupy ndarray exists in the arguments
            return self.func(*args)

        # Check for invalid argument types
        for i in range(nargs):
            if not isinstance(args[i], _fusion_argument_types):
                mes = 'Invalid argument type for \'{}\': ({})'
                arg_types = ', '.join(repr(type(a)) for a in args)
                raise TypeError(mes.format(self.name, arg_types))

        # Cache the result of execution path trace
        # TODO(asi1024): Some ndarrays may share the same memory space.
        cdef list params_info = []
        cdef list shape_info = []

        # Create cache keys to find a kernel already emitted:
        #     param_key: ndims and dtypes of the arguments.
        #     shape_key: shapes of the arguments.
        for i in range(nargs):
            arg = args[i]
            if isinstance(arg, core.ndarray):
                params_info.append(arg.dtype.char)
                params_info.append(arg.ndim)
                shape_info.append(arg.shape)
                base = arg.base
                params_info.append(base is None)
                for j in range(i):
                    params_info.append(arg is args[j])
                    if base is not None and isinstance(args[j], core.ndarray):
                        # args[i] and args[j] may share the same memory space.
                        params_info.append(base is args[j].base)
            elif isinstance(arg, numpy.generic):
                params_info.append(arg.dtype.char)
            elif arg is None:
                params_info.append(None)
            elif isinstance(arg, float):
                params_info.append('d')
            elif isinstance(arg, int):
                params_info.append('l')
            elif isinstance(arg, bool):
                params_info.append('?')
            elif isinstance(arg, complex):
                params_info.append('D')
            else:
                raise TypeError('Unsupported input type {}.'.format(type(arg)))

        cdef tuple param_key = tuple(params_info)
        cdef tuple shape_key = tuple(shape_info)

        # self._cache(dict):
        #     key:   Tuple of dtypes and ndims of the inputs (param_key).
        #     value: Pair of cache_shape and kernel_list.
        # cache_shape(dict):
        #     key:   Tuple of shapes of the inputs (shape_key).
        #     value: Pair of Runtime kernel and its performance cookie.
        # kernel_list(list): List of Runtime kernel.
        cache_shape, kernel_list = self._cache.get(param_key, (None, None))

        if cache_shape is None:
            # Initializes cache_shape and kernel_list.
            cache_shape = {}
            kernel_list = []
            self._cache[param_key] = cache_shape, kernel_list

        # Find a cached kernel from cache_shape with the key as actual shape.
        kernel, shapes = cache_shape.get(shape_key, (None, None))

        if kernel is None:
            # Find a cached kernel from kernel_list.

            # Create a dim_map: a dictionary from _AbstractDim to int.
            dim_map = {}
            for input_index, arg in enumerate(args):
                if isinstance(arg, core.ndarray):
                    for axis, dim in enumerate(arg.shape):
                        dim_map[_AbstractDim(input_index, axis)] = dim

            # Find a kernel that satisfies the shape constraints.
            for cand_kernel in kernel_list:
                if cand_kernel.shape_constraints.satisfy(dim_map):
                    kernel = cand_kernel
                    shapes = kernel.get_shapes_of_kernel_params(args)
                    cache_shape[shape_key] = kernel, shapes
                    break

        if kernel is None:
            # If not cached at all, analyze the target function.
            kernel = _get_fused_kernel(self.name, self.func, args)
            shapes = kernel.get_shapes_of_kernel_params(args)
            cache_shape[shape_key] = kernel, shapes
            kernel_list.append(kernel)

        return kernel.execute(args, shapes)


def fuse(*args, **kwargs):
    """Decorator that fuses a function.
    """

    def wrapper(f, kernel_name=None):
        return Fusion(f, kernel_name)

    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return functools.update_wrapper(wrapper(args[0]), args[0])
    else:
        return lambda f: functools.update_wrapper(
            wrapper(f, *args, **kwargs), f)
