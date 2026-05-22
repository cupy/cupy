import re
import cupy
import numpy as np

from cupy import _util
from cupy._core._kernel import _get_param_info

from cupy.cuda cimport device

from cupy._core cimport _carray
from cupy._core cimport _scalar
from cupy._core.core cimport _ndarray_base
from cupy._core._carray cimport shape_t
from cupy._core._kernel cimport _preprocess_args
from cupy._core._kernel cimport ParameterInfo
from cupy._core._kernel cimport _decide_params_type_core
from cupy._core._kernel cimport _get_arg_infos
from cupy._core._kernel cimport _get_simple_elementwise_kernel_code
from cupy._core._kernel cimport _get_simple_elementwise_kernel_from_code
from cupy._core._kernel cimport _XwiseKernelBase


cdef inline bint _contains_zero(const shape_t& v) except? -1:
    for i in range(v.size()):
        if v[i] == 0:
            return True
    return False


@_util.memoize()
def _get_batchwise_param_info(str s, bint is_const):
    if not s:
        return (), ()

    params = []
    core_ndims = []

    for param in s.split(','):
        param = param.strip()
        if not param:
            continue

        parts = param.split()
        if len(parts) < 2:
            raise ValueError(f"Syntax error: {param}")

        type_str = parts[-2]

        sub_parts = type_str.rsplit('_', maxsplit=1)
        if (
                len(sub_parts) == 2
                and sub_parts[1].endswith('d')
                and sub_parts[1][:-1].isdigit()
        ):
            parts[-2] = sub_parts[0]
            core_ndim = int(sub_parts[1][:-1])
        else:
            core_ndim = 0

        params.append(ParameterInfo(' '.join(parts), is_const))
        core_ndims.append(core_ndim)

    return tuple(params), tuple(core_ndims)


@_util.memoize()
def _get_batchwise_kernel_code(
        tuple arginfos, object type_map,
        tuple params, str operation, str name,
        str preamble, str loop_prep='', str after_loop='', tuple options=()):
    cdef _ArgInfo arginfo

    op = []
    for p, arginfo in zip(params, arginfos):
        if arginfo.is_ndarray() and not p.raw:
            if arginfo.core_ndim > 0:
                if p.is_const:
                    fmt = 'const auto {n} = _raw_{n}[_ind.get()];'
                else:
                    fmt = 'auto {n} = _raw_{n}[_ind.get()];'
            else:
                if p.is_const:
                    fmt = 'const {t} &{n} = _raw_{n}[_ind.get()];'
                else:
                    fmt = '{t} &{n} = _raw_{n}[_ind.get()];'
                    
            op.append(fmt.format(t=p.ctype, n=p.name))
            
    op.append(operation)
    operation = '\n'.join(op)

    return _get_simple_elementwise_kernel_code(
        params, arginfos, operation, name, type_map,
        preamble, loop_prep, after_loop)


@_util.memoize(for_each_device=True)
def _get_batchwise_kernel(
        tuple arginfos, object type_map,
        tuple params, str operation, str name,
        str preamble, str loop_prep='', str after_loop='', tuple options=()):
        
    cdef str code = _get_batchwise_kernel_code(
        arginfos, type_map, params, operation, name, preamble, loop_prep,
        after_loop
    )
    # Hand the C++ string to NVRTC to compile into a binary
    return _get_simple_elementwise_kernel_from_code(name, code, options)


cdef class BatchwiseKernel(_XwiseKernelBase):
    """docstring will go here."""
    cdef:
        readonly tuple in_core_ndims
        readonly tuple out_core_ndims
        readonly object validate_core_shapes

    def __init__(self, in_params, out_params, operation,
                 name='kernel', preamble='', *, validate_core_shapes):
        
        self.in_params, self.in_core_ndims = _get_batchwise_param_info(
            in_params, True)
        self.out_params, self.out_core_ndims = _get_batchwise_param_info(
            out_params, False)
        self.nin = len(self.in_params)
        self.nout = len(self.out_params)
        self.nargs = self.nin + self.nout
        param_rest = _get_param_info('CIndexer _ind', False)
        self.params = self.in_params + self.out_params + param_rest
        self.operation = operation
        self.preamble = preamble
        self.validate_core_shapes = validate_core_shapes
        self._params_type_memo = {}
        self._cached_codes = {}
        names = [p.name for p in self.in_params + self.out_params]
        if 'i' in names:
            raise ValueError('Can not use \'i\' as a parameter name.')
        self._kernel_memo = {}
        # This is for profiling mechanisms to auto infer a name.
        self.__name__ = name
            
    def __call__(self, *args, out):
        cdef list in_args, out_args
        cdef tuple in_types, out_types
        cdef shape_t loop_shape

        if len(args) != self.nin:
            raise TypeError(
                f"Wrong number of arguments for {self.name}. "
                f"Expected {self.nin}, got {len(args)}."
            )

        if not isinstance(out, list):
            out = [out]

        if len(out) != self.nout:
            raise TypeError(
                f"Wrong number of outputs for {self.name}. "
                f"Expected {self.nout}, got {len(out)}."
            )
    
        dev_id = device.get_device_id()
        in_args = _preprocess_args(dev_id, args)
        out_args = _preprocess_args(dev_id, out)

        if any(
            [
                (arg.ndim if isinstance(arg, _ndarray_base) else 0)
                < self.in_core_ndims[i]
                for i, arg in enumerate(in_args)
            ]
        ):
            raise ValueError("input with insufficient dimensions.")

        if any(
            [
                (arg.ndim if isinstance(arg, _ndarray_base) else 0)
                < self.out_core_ndims[i]
                for i, arg in enumerate(out_args)
            ]
        ):
            raise ValueError("output with insufficient dimensions.")

        in_core_shapes = tuple(
            arg.shape[-self.in_core_ndims[i]:]
            if self.in_core_ndims[i] > 0 else ()
            for i, arg in enumerate(in_args)
        )
        out_core_shapes = tuple(
            arg.shape[-self.out_core_ndims[i]:]
            if self.out_core_ndims[i] > 0 else ()
            for i, arg in enumerate(out_args)
        )
        self.validate_core_shapes(in_core_shapes, out_core_shapes)

        in_batch_shapes = [
            arg.shape[:-self.in_core_ndims[i]] if self.in_core_ndims[i] > 0
            else arg.shape if isinstance(arg, _ndarray_base) else ()
            for i, arg in enumerate(in_args)
        ]

        out_batch_shapes = [
            arg.shape[:-self.out_core_ndims[i]] if self.out_core_ndims[i] > 0 
            else arg.shape if isinstance(arg, _ndarray_base) else ()
            for i, arg in enumerate(out_args)
        ]
        batch_shape = np.broadcast_shapes(*in_batch_shapes, *out_batch_shapes)
        if any(shape != batch_shape for shape in out_batch_shapes):
            raise ValueError("output batch shape mismatch.")

        in_args = [
            cupy.broadcast_to(arg, batch_shape + core_shape)
            if isinstance(arg, _ndarray_base) else arg
            for arg, core_shape in zip(in_args, in_core_shapes)
        ]

        in_ndarray_types = []
        for a in in_args:
            if isinstance(a, _ndarray_base):
                t = a.dtype
            elif isinstance(a, texture.TextureObject):
                t = 'cudaTextureObject_t'
            else:
                t = None
            in_ndarray_types.append(t)
        in_ndarray_types = tuple(in_ndarray_types)
        out_ndarray_types = tuple([a.dtype for a in out_args])

        in_types, out_types, type_map = self._decide_params_type(
            in_ndarray_types, out_ndarray_types)

        loop_shape = batch_shape
        if _contains_zero(loop_shape):
            return out if len(out) > 1 else out[0]

        for i, x in enumerate(in_args):
            if type(x) is _scalar.CScalar:
                (<_scalar.CScalar>x).apply_dtype(in_types[i])

        inout_args = in_args + out_args

        indexer = _carray._indexer_init(loop_shape)
        inout_args.append(indexer)

        core_ndims = self.in_core_ndims + self.out_core_ndims
        arginfos = _get_arginfos(inout_args, core_ndims=core_ndims)
        kern = self._get_kernel(dev_id, arginfos, type_map)

        kern.linear_launch(indexer.size, inout_args, shared_mem=0,
                           block_max_size=0, stream=None)
        return out if self.nout > 1 else out[0]

    cdef function.Function _compile_kernel(
            self, int dev_id, tuple arginfos, object type_map):
        return _get_batchwise_kernel(
            arginfos, type_map, self.params, self.operation,
            self.name, self.preamble)

    cdef str _get_kernel_code(self, tuple arginfos, object type_map):
        return _get_batchwise_kernel_code(
            arginfos, type_map, self.params, self.operation,
            self.name, self.preamble)
