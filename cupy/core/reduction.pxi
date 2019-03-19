from cpython cimport sequence
from libc.stdint cimport int32_t

from cupy.core cimport _routines_manipulation as _manipulation

import string

import numpy

from cupy.core import _errors
from cupy.cuda import compiler
from cupy import util


cpdef _get_simple_reduction_kernel(
        name, block_size, reduce_type, params, identity,
        pre_map_expr, reduce_expr, post_map_expr,
        type_preamble, input_expr, output_expr, preamble, options):
    if identity is None:
        identity = ''
    module_code = string.Template('''
${type_preamble}
${preamble}
#define REDUCE(a, b) (${reduce_expr})
#define POST_MAP(a) (${post_map_expr})
#define _REDUCE(_offset) if (_tid < _offset) { \
  _type_reduce _a = _sdata[_tid], _b = _sdata[(_tid + _offset)]; \
  _sdata[_tid] = REDUCE(_a, _b); \
}

typedef ${reduce_type} _type_reduce;
extern "C" __global__ void ${name}(${params}) {
  __shared__ char _sdata_raw[${block_size} * sizeof(_type_reduce)];
  _type_reduce *_sdata = reinterpret_cast<_type_reduce*>(_sdata_raw);
  unsigned int _tid = threadIdx.x;

  int _J_offset = _tid >> __popc(_block_stride - 1);  // _tid / _block_stride
  ptrdiff_t _j_offset = (ptrdiff_t)_J_offset * _out_ind.size();
  int _J_stride = ${block_size} >> __popc(_block_stride - 1);
  ptrdiff_t _j_stride = (ptrdiff_t)_J_stride * _out_ind.size();

  for (ptrdiff_t _i_base = (ptrdiff_t)blockIdx.x * _block_stride;
       _i_base < _out_ind.size();
       _i_base += (ptrdiff_t)gridDim.x * _block_stride) {
    _type_reduce _s = _type_reduce(${identity});
    ptrdiff_t _i =
        _i_base + (_tid & (_block_stride - 1));  // _tid % _block_stride
    int _J = _J_offset;
    for (ptrdiff_t _j = _i + _j_offset; _j < _in_ind.size();
         _j += _j_stride, _J += _J_stride) {
      _in_ind.set(_j);
      ${input_expr}
      _type_reduce _a = static_cast<_type_reduce>(${pre_map_expr});
      _s = REDUCE(_s, _a);
    }
    _sdata[_tid] = _s;
    __syncthreads();
    for (unsigned int _block = ${block_size} / 2;
         _block >= _block_stride; _block >>= 1) {
      if (_tid < _block) {
        _REDUCE(_block);
      }
      __syncthreads();
    }
    if (_tid < _block_stride) {
      _s = _sdata[_tid];
    }
    if (_tid < _block_stride && _i < _out_ind.size()) {
      _out_ind.set(static_cast<ptrdiff_t>(_i));
      ${output_expr}
      POST_MAP(_s);
    }
  }
}''').substitute(
        name=name,
        block_size=block_size,
        reduce_type=reduce_type,
        params=params,
        identity=identity,
        reduce_expr=reduce_expr,
        pre_map_expr=pre_map_expr,
        post_map_expr=post_map_expr,
        type_preamble=type_preamble,
        input_expr=input_expr,
        output_expr=output_expr,
        preamble=preamble)
    module = compile_with_cache(module_code, options)
    return module.get_function(name)


cpdef tuple _get_axis(object axis, Py_ssize_t ndim):
    cdef Py_ssize_t dim
    if axis is None:
        axis = tuple(range(ndim))
    elif sequence.PySequence_Check(axis):
        axis = tuple(axis)
    else:
        axis = axis,

    for dim in axis:
        if dim < -ndim or dim >= ndim:
            raise _errors._AxisError('Axis overrun')
    reduce_axis = tuple(sorted([dim % ndim for dim in axis]))
    out_axis = tuple([dim for dim in range(ndim) if dim not in reduce_axis])
    return reduce_axis, out_axis


cpdef tuple _get_out_shape(
        tuple shape, tuple reduce_axis, tuple out_axis, bint keepdims):
    if keepdims:
        out_shape = list(shape)
        for i in reduce_axis:
            out_shape[i] = 1
        return tuple(out_shape)
    return tuple([shape[i] for i in out_axis])


cpdef tuple _get_permuted_args(
        list args, tuple axis_permutes, tuple shape, tuple params,
        Py_ssize_t out_ndim):
    cdef ParameterInfo p
    cdef Py_ssize_t contiguous_size, tmp_contiguous_size, itemsize

    if axis_permutes != tuple(range(len(shape))):
        if params is not None:
            for p in params:
                if p.raw:
                    raise NotImplementedError('Illegal conditions')
        args = [_manipulation._transpose(a, axis_permutes)
                if isinstance(a, ndarray) else a for a in args]
        shape = tuple([shape[i] for i in axis_permutes])

    contiguous_size = 1
    for a in args:
        if not isinstance(a, ndarray):
            continue
        tmp_contiguous_size = 1
        itemsize = a.dtype.itemsize
        for i in range(out_ndim):
            if a.strides[-i-1] != tmp_contiguous_size * itemsize:
                break
            tmp_contiguous_size *= a.shape[-i-1]
        contiguous_size = max(contiguous_size, tmp_contiguous_size)
    return args, shape, contiguous_size


cpdef (Py_ssize_t, Py_ssize_t, Py_ssize_t) _get_block_specs(  # NOQA
        Indexer in_indexer, Indexer out_indexer, Py_ssize_t contiguous_size):
    cdef Py_ssize_t block_size, reduce_block_size, block_stride, out_block_num
    block_size = 512

    reduce_block_size = max(1, in_indexer.size // out_indexer.size)
    contiguous_size = min(contiguous_size, 32)
    block_stride = max(contiguous_size, block_size // reduce_block_size)
    block_stride = internal.clp2(block_stride // 2 + 1)  # floor
    out_block_num = (out_indexer.size + block_stride - 1) // block_stride

    return block_size, block_stride, out_block_num


cpdef list _get_inout_args(
        list in_args, list out_args, Indexer in_indexer, Indexer out_indexer,
        Py_ssize_t block_stride, tuple params, bint reduce_dims):
    if reduce_dims:
        in_shape = _reduce_dims(in_args, params, in_indexer.shape)
        out_shape = _reduce_dims(
            out_args, params[len(in_args):], out_indexer.shape)
        in_indexer.shape = in_shape
        out_indexer.shape = out_shape
    cdef _scalar.CScalar s = _scalar.CScalar.__new__(_scalar.CScalar)
    (<int32_t *>s.ptr)[0] = block_stride
    s.kind = b'i'
    s.size = 4
    return in_args + out_args + [in_indexer, out_indexer, s]


@util.memoize(for_each_device=True)
def _get_simple_reduction_function(
        routine, params, args_info, in_arg_dtype, out_arg_dtype, out_types,
        name, block_size, identity, input_expr, output_expr, _preamble,
        options):
    reduce_type = routine[3]
    if reduce_type is None:
        reduce_type = _get_typename(out_types[0])

    t = (_get_typename(in_arg_dtype), _get_typename(out_arg_dtype))
    type_preamble = 'typedef %s type_in0_raw; typedef %s type_out0_raw;' % t

    params = _get_kernel_params(params, args_info)
    return _get_simple_reduction_kernel(
        name, block_size, reduce_type, params, identity,
        routine[0], routine[1], routine[2],
        type_preamble, input_expr, output_expr, _preamble, options)


class simple_reduction_function(object):

    def __init__(self, name, ops, identity, preamble):
        self.name = name
        self._ops = ops
        self.identity = identity
        self._preamble = preamble
        self.nin = 1
        self.nout = 1
        in_params = _get_param_info('T in0', True)
        out_params = _get_param_info('T out0', False)
        self._params = (
            in_params + out_params +
            _get_param_info('CIndexer _in_ind, CIndexer _out_ind', False) +
            _get_param_info('int32 _block_stride', True))
        self._input_expr = 'const type_in0_raw in0 = _raw_in0[_in_ind.get()];'
        self._output_expr = 'type_out0_raw &out0 = _raw_out0[_out_ind.get()];'
        self._routine_cache = {}

    def __call__(self, object a, axis=None, dtype=None, ndarray out=None,
                 bint keepdims=False):
        cdef list in_args, out_args
        cdef tuple in_sahpe, reduce_axis, out_axis
        cdef Py_ssize_t contiguous_size
        cdef Py_ssize_t block_size, block_stride, out_block_num
        cdef ndarray arr
        if dtype is not None:
            dtype = get_dtype(dtype).type

        if isinstance(a, ndarray):
            arr = a
        elif hasattr(a, '__cuda_array_interface__'):
            arr = _convert_object_with_cuda_array_interface(a)
        else:
            raise TypeError(
                'Argument \'a\' has incorrect type (expected %s, got %s)' %
                (ndarray, type(a)))
        del a
        in_args = [arr]
        a_shape = arr.shape
        dev_id = device.get_device_id()
        if out is None:
            _preprocess_args(dev_id, (arr,), False)
            out_args = []
        else:
            _preprocess_args(dev_id, (arr, out), False)
            out_args = [out]

        in_types, out_types, routine = _guess_routine(
            self.name, self._routine_cache, self._ops, in_args, dtype)

        reduce_axis, out_axis = _get_axis(axis, arr._shape.size())
        del axis  # to avoid bug
        out_shape = _get_out_shape(a_shape, reduce_axis, out_axis, keepdims)
        out_args = _get_out_args(out_args, out_types, out_shape, 'unsafe')
        ret = out_args[0] if len(out_args) == 1 else tuple(out_args)
        if (<ndarray>out_args[0]).size == 0:
            return ret
        if arr.size == 0 and self.identity is None:
            raise ValueError(('zero-size array to reduction operation'
                              ' %s which has no identity') % self.name)

        in_args, in_shape, contiguous_size = _get_permuted_args(
            in_args, reduce_axis + out_axis, a_shape, None, len(out_axis))

        in_indexer = Indexer(in_shape)
        out_indexer = Indexer(out_shape)
        block_size, block_stride, out_block_num = _get_block_specs(
            in_indexer, out_indexer, contiguous_size)

        inout_args = _get_inout_args(
            in_args, out_args, in_indexer, out_indexer, block_stride,
            self._params, True)
        args_info = _get_args_info(inout_args)

        kern = _get_simple_reduction_function(
            routine, self._params, args_info,
            in_args[0].dtype.type, out_args[0].dtype.type, out_types,
            self.name, block_size, self.identity,
            self._input_expr, self._output_expr, self._preamble, ())
        kern.linear_launch(
            out_block_num * block_size, inout_args, 0, block_size)
        return ret


@util.memoize(for_each_device=True)
def _get_reduction_kernel(
        nin, nout, params, args_info, types,
        name, block_size, reduce_type, identity, map_expr, reduce_expr,
        post_map_expr, preamble, options):
    kernel_params = _get_kernel_params(params, args_info)
    params = params[:nin + nout]
    args_info = args_info[:nin + nout]
    in_arrays = [p for p, a in zip(params[:nin], args_info[:nin])
                 if not p.raw and a[0] is ndarray]
    out_arrays = [p for p, a in zip(params[nin:], args_info[nin:])
                  if not p.raw and a[0] is ndarray]
    type_preamble = '\n'.join(
        'typedef %s %s;' % (_get_typename(v), k)
        for k, v in types)
    input_expr = '\n'.join(
        [(('const {0} {1}' if p.is_const else '{0}& {1}') +
          ' = _raw_{1}[_in_ind.get()];').format(p.ctype, p.name)
         for p in in_arrays])
    output_expr = '\n'.join(
        ['{0} &{1} = _raw_{1}[_out_ind.get()];'.format(p.ctype, p.name)
         for p in out_arrays if not p.is_const])

    return _get_simple_reduction_kernel(
        name, block_size, reduce_type, kernel_params, identity,
        map_expr, reduce_expr, post_map_expr,
        type_preamble, input_expr, output_expr, preamble, options)


class ReductionKernel(object):

    """User-defined reduction kernel.

    This class can be used to define a reduction kernel with or without
    broadcasting.

    The kernel is compiled at an invocation of the
    :meth:`~ReductionKernel.__call__` method, which is cached for each device.
    The compiled binary is also cached into a file under the
    ``$HOME/.cupy/kernel_cache/`` directory with a hashed file name. The cached
    binary is reused by other processes.

    Args:
        in_params (str): Input argument list.
        out_params (str): Output argument list.
        map_expr (str): Mapping expression for input values.
        reduce_expr (str): Reduction expression.
        post_map_expr (str): Mapping expression for reduced values.
        identity (str): Identity value for starting the reduction.
        name (str): Name of the kernel function. It should be set for
            readability of the performance profiling.
        reduce_type (str): Type of values to be used for reduction. This type
            is used to store the special variables ``a``.
        reduce_dims (bool): If ``True``, input arrays are reshaped without copy
            to smaller dimensions for efficiency.
        preamble (str): Fragment of the CUDA-C/C++ code that is inserted at the
            top of the cu file.
        options (tuple of str): Additional compilation options.

    """
    def __init__(self, in_params, out_params,
                 map_expr, reduce_expr, post_map_expr,
                 identity, name='reduce_kernel', reduce_type=None,
                 reduce_dims=True, preamble='', options=()):
        if not compiler.is_valid_kernel_name(name):
            raise ValueError(
                'Invalid kernel name: "%s"' % name)

        self.in_params = _get_param_info(in_params, True)
        self.out_params = _get_param_info(out_params, False)
        self.nin = len(self.in_params)
        self.nout = len(self.out_params)
        self.nargs = self.nin + self.nout
        self.params = (
            self.in_params + self.out_params +
            _get_param_info('CIndexer _in_ind, CIndexer _out_ind', False) +
            _get_param_info('int32 _block_stride', True))
        self.identity = identity
        self.reduce_expr = reduce_expr
        self.map_expr = map_expr
        self.name = name
        self.options = options
        self.reduce_dims = reduce_dims
        self.post_map_expr = post_map_expr
        if reduce_type is None:
            self.reduce_type = self.out_params[0].ctype
        else:
            self.reduce_type = reduce_type
        self.preamble = preamble

    def __call__(self, *args, **kwargs):
        """Compiles and invokes the reduction kernel.

        The compilation runs only if the kernel is not cached. Note that the
        kernels with different argument dtypes, ndims, or axis are not
        compatible. It means that single ReductionKernel object may be compiled
        into multiple kernel binaries.

        Args:
            args: Arguments of the kernel.
            axis (int or tuple of ints): Axis or axes along which the
                reduction is performed.
            keepdims (bool): If ``True``, the specified axes are remained as
                axes of length one.

        Returns:
            Arrays are returned according to the ``out_params`` argument of the
            ``__init__`` method.

        """
        cdef Py_ssize_t contiguous_size
        cdef Py_ssize_t block_size, block_stride, out_block_num

        out = kwargs.pop('out', None)
        axis = kwargs.pop('axis', None)
        keepdims = kwargs.pop('keepdims', False)
        stream = kwargs.pop('stream', None)
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)

        n_args = len(args)
        if n_args != self.nin and n_args != self.nargs:
            raise TypeError('Wrong number of arguments for %s' % self.name)

        out_args = list(args[self.nin:])
        if out is not None:
            if self.nout != 1:
                raise NotImplementedError('')
            if len(out_args) != 0:
                raise ValueError("cannot specify 'out' as both "
                                 "a positional and keyword argument")
            out_args = [out]

        dev_id = device.get_device_id()
        in_args = _preprocess_args(dev_id, args[:self.nin], False)
        out_args = _preprocess_args(dev_id, out_args, False)
        in_args, broad_shape = _broadcast(in_args, self.in_params, False)

        if self.identity is None and 0 in broad_shape:
            raise ValueError(('zero-size array to reduction operation'
                              ' %s which has no identity') % self.name)

        in_ndarray_types = tuple(
            [a.dtype.type if isinstance(a, ndarray) else None
             for a in in_args])
        out_ndarray_types = tuple(
            [a.dtype.type if isinstance(a, ndarray) else None
             for a in out_args])
        in_types, out_types, types = _decide_params_type(
            self.in_params, self.out_params,
            in_ndarray_types, out_ndarray_types)

        reduce_axis, out_axis = _get_axis(axis, len(broad_shape))
        out_shape = _get_out_shape(
            broad_shape, reduce_axis, out_axis, keepdims)
        out_args = _get_out_args_with_params(
            out_args, out_types, out_shape, self.out_params, False)
        ret = out_args[0]
        if 0 in out_shape:
            return ret

        in_args = [x if isinstance(x, ndarray) else
                   _scalar.get_scalar_from_numpy(x, t)
                   for x, t in zip(in_args, in_types)]
        in_args, in_shape, contiguous_size = _get_permuted_args(
            in_args, reduce_axis + out_axis, broad_shape, self.in_params,
            len(out_axis))

        in_indexer = Indexer(in_shape)
        out_indexer = Indexer(out_shape)
        block_size, block_stride, out_block_num = _get_block_specs(
            in_indexer, out_indexer, contiguous_size)

        inout_args = _get_inout_args(
            in_args, out_args, in_indexer, out_indexer, block_stride,
            self.params, self.reduce_dims)
        args_info = _get_args_info(inout_args)

        kern = _get_reduction_kernel(
            self.nin, self.nout, self.params, args_info, types,
            self.name, block_size, self.reduce_type, self.identity,
            self.map_expr, self.reduce_expr, self.post_map_expr,
            self.preamble, self.options)
        kern.linear_launch(
            out_block_num * block_size, inout_args, 0, block_size, stream)
        return ret


cpdef create_reduction_func(name, ops, routine=None, identity=None,
                            preamble=''):
    _ops = []
    for t in ops:
        if not isinstance(t, tuple):
            typ = t
            rt = routine
        else:
            typ, rt = t
            rt = tuple([i or j for i, j in zip(rt, routine)])

        types = typ.split('->')
        if len(types) == 1:
            in_types = out_types = tuple(types)
        else:
            in_types, out_types = map(tuple, types)
        in_types = tuple([get_dtype(t).type for t in in_types])
        out_types = tuple([get_dtype(t).type for t in out_types])
        _ops.append((in_types, out_types, rt))

    return simple_reduction_function(name, _ops, identity, preamble)
