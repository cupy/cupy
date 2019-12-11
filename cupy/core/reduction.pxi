from cpython cimport sequence

from cupy.core cimport _routines_manipulation as _manipulation
from cupy.cuda cimport function
from cupy.cuda cimport runtime

import string

from cupy.core import _errors
from cupy.cuda import compiler
from cupy import util


cpdef function.Function _create_reduction_function(
        name, block_size, reduce_type, params, identity,
        pre_map_expr, reduce_expr, post_map_expr,
        type_preamble, input_expr, output_expr, preamble, options,
        is_c_contiguous):
    if is_c_contiguous:
        thread_row_expr = 'threadIdx.x >> __popc(_block_stride - 1)'
        thread_col_expr = 'threadIdx.x & (_block_stride - 1)'
    else:
        thread_row_expr = 'threadIdx.x & (_J_stride - 1)'
        thread_col_expr = 'threadIdx.x >> __popc(_J_stride - 1)'

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

  int _J_stride = ${block_size} >> __popc(_block_stride - 1);
  ptrdiff_t _j_stride = (ptrdiff_t)_J_stride * _out_ind.size();

  // _thread_row = _tid / _block_stride
  // _thread_col = _tid % _block_stride
  unsigned int _thread_row = ${thread_row_expr};
  unsigned int _thread_col = ${thread_col_expr};
  unsigned int _tid = _thread_row * _block_stride + _thread_col;

  int _J_offset = _thread_row;
  ptrdiff_t _j_offset = (ptrdiff_t)_J_offset * _out_ind.size();

  for (ptrdiff_t _i_base = (ptrdiff_t)blockIdx.x * _block_stride;
       _i_base < _out_ind.size();
       _i_base += (ptrdiff_t)gridDim.x * _block_stride) {
    _type_reduce _s = _type_reduce(${identity});
    ptrdiff_t _i = _i_base + _thread_col;
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
    if (threadIdx.x < _block_stride) {
      _s = _sdata[threadIdx.x];
    }
    if (threadIdx.x < _block_stride && _i < _out_ind.size()) {
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
        thread_row_expr=thread_row_expr,
        thread_col_expr=thread_col_expr,
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


cdef tuple _set_permuted_args(
        list args, tuple axis_permutes, tuple shape, tuple params):
    # This function updates `args`
    cdef ParameterInfo p
    cdef Py_ssize_t i, s
    cdef bint need_permutation = False
    for i, s in enumerate(axis_permutes):
        if i != s:
            need_permutation = True
            break
    if need_permutation:
        for p in params:
            if p.raw:
                raise NotImplementedError('Illegal conditions')
        for i, a in enumerate(args):
            if isinstance(a, ndarray):
                args[i] = _manipulation._transpose(a, axis_permutes)
        shape = tuple([shape[i] for i in axis_permutes])
    return shape


cdef Py_ssize_t _get_contiguous_size(
        list args, tuple params, Py_ssize_t ndim,
        Py_ssize_t out_ndim) except -1:
    cdef int i, j
    cdef ParameterInfo p
    cdef Py_ssize_t contiguous_size, tmp_contiguous_size, itemsize
    contiguous_size = 1
    for i, a in enumerate(args):
        if not isinstance(a, ndarray):
            continue
        p = params[i]
        if p.raw:
            continue
        tmp_contiguous_size = 1
        itemsize = a.dtype.itemsize
        for j in range(out_ndim):
            if a._strides[ndim-j-1] != tmp_contiguous_size * itemsize:
                break
            tmp_contiguous_size *= a._shape[ndim-j-1]
        contiguous_size = max(contiguous_size, tmp_contiguous_size)
    return contiguous_size


cpdef (Py_ssize_t, Py_ssize_t, Py_ssize_t, bint) _get_block_specs(  # NOQA
        Py_ssize_t in_size, Py_ssize_t out_size,
        Py_ssize_t contiguous_size) except*:
    cdef Py_ssize_t reduce_block_size, block_stride, out_block_num

    reduce_block_size = max(1, in_size // out_size)
    contiguous_size = min(contiguous_size, 32)
    block_stride = max(contiguous_size, _block_size // reduce_block_size)
    block_stride = internal.clp2(block_stride // 2 + 1)  # floor
    out_block_num = (out_size + block_stride - 1) // block_stride

    return _block_size, block_stride, out_block_num, contiguous_size > 1


cdef Py_ssize_t _block_size = 256 if runtime._is_hip_environment else 512


cdef tuple _get_reduction_args(
        list in_args, list out_args, tuple in_params, tuple out_params,
        tuple axis_permutes, tuple a_shape, tuple out_shape,
        bint reduce_dims):
    # Returns a tuple that contains following items
    # - list of arguments passed to the __global__ function.
    # - block_size
    # - out_block_num
    cdef Py_ssize_t contiguous_size, block_size, block_stride, out_block_num
    in_shape = _set_permuted_args(
        in_args, axis_permutes, a_shape, in_params)
    contiguous_size = _get_contiguous_size(
        in_args, in_params, len(in_shape), len(out_shape))

    if reduce_dims:
        in_shape = _reduce_dims(in_args, in_params, in_shape)
        out_shape = _reduce_dims(out_args, out_params, out_shape)

    block_size, block_stride, out_block_num, c_contiguous = _get_block_specs(
        internal.prod_sequence(in_shape),
        internal.prod_sequence(out_shape),
        contiguous_size)

    in_indexer = Indexer(in_shape)
    out_indexer = Indexer(out_shape)

    # The last argument is always block_stride.
    s = _scalar.CScalar_from_int32(block_stride)
    return (in_args + out_args + [in_indexer, out_indexer, s],
            block_size, out_block_num, c_contiguous)


cdef class _AbstractReductionKernel:

    cdef:
        readonly str name
        public str identity
        readonly tuple in_params
        readonly tuple out_params
        readonly tuple _params

    def __init__(
            self, str name, str identity, str in_params, str out_params):
        assert name is not None
        assert identity is not None
        assert in_params is not None
        assert out_params is not None

        in_params_ = _get_param_info(in_params, True)
        out_params_ = _get_param_info(out_params, False)
        params = (
            in_params_
            + out_params_
            + _get_param_info('CIndexer _in_ind, CIndexer _out_ind', False)
            + _get_param_info('int32 _block_stride', True))

        self.name = name
        self.identity = identity
        self.in_params = in_params_
        self.out_params = out_params_
        self._params = params

    cpdef ndarray _call(
            self,
            list in_args, list out_args,
            tuple a_shape, axis, dtype,
            bint keepdims, bint reduce_dims,
            stream):
        cdef tuple reduce_axis, out_axis
        cdef Py_ssize_t contiguous_size
        cdef Py_ssize_t block_size, block_stride, out_block_num
        cdef ndarray ret
        cdef function.Function kern

        if dtype is not None:
            dtype = get_dtype(dtype).type

        (
            map_expr, reduce_expr, post_map_expr,
            in_types, out_types, reduce_type,
            types,
        ) = self._get_expressions_and_types(in_args, out_args, dtype)

        reduce_axis, out_axis = _get_axis(axis, len(a_shape))
        out_shape = _get_out_shape(a_shape, reduce_axis, out_axis, keepdims)
        out_args = self._get_out_args(out_args, out_types, out_shape)
        ret = out_args[0]
        if ret.size == 0:
            return ret

        if self.identity == '' and 0 in a_shape:
            raise ValueError(('zero-size array to reduction operation'
                              ' %s which has no identity') % self.name)

        in_args = [x if isinstance(x, ndarray) else
                   _scalar.get_scalar_from_numpy(x, t)
                   for x, t in zip(in_args, in_types)]
        inout_args, block_size, out_block_num, is_c_contiguous = _get_reduction_args(
            in_args, out_args, self.in_params, self.out_params,
            reduce_axis + out_axis, a_shape, out_shape, reduce_dims)
        args_info = _get_args_info(inout_args)

        func = self._get_function(
            self._params, args_info, types,
            map_expr, reduce_expr, post_map_expr, reduce_type,
            block_size, is_c_contiguous)
        func.linear_launch(
            out_block_num * block_size, inout_args, 0, block_size, stream)
        return ret

    cdef tuple _get_expressions_and_types(
            self, list in_args, list out_args, dtype):
        raise NotImplementedError()

    cdef list _get_out_args(
            self, list out_args, tuple out_types, tuple out_shape):
        raise NotImplementedError()

    cdef function.Function _get_function(
            self,
            tuple params, tuple args_info, tuple types,
            str map_expr, str reduce_expr, str post_map_expr, str reduce_type,
            Py_ssize_t block_size, bint is_c_contiguous):
        raise NotImplementedError()


# -----------------------------------------------------------------------------
# create_reduction_func
# -----------------------------------------------------------------------------

cpdef _SimpleReductionKernel create_reduction_func(
        name, ops, routine=None, identity=None, preamble=''):
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

    return _SimpleReductionKernel(name, _ops, identity, preamble)


cdef class _SimpleReductionKernel(_AbstractReductionKernel):

    cdef:
        readonly object _ops
        readonly _preamble
        readonly int nin
        readonly int nout
        readonly str _input_expr
        readonly str _output_expr
        readonly dict _routine_cache

    def __init__(self, name, ops, identity, preamble):
        super().__init__(
            name,
            '' if identity is None else str(identity),
            'T in0',
            'T out0',
        )
        self._ops = ops
        self._preamble = preamble
        self.nin = 1
        self.nout = 1
        self._input_expr = 'const type_in0_raw in0 = _raw_in0[_in_ind.get()];'
        self._output_expr = 'type_out0_raw &out0 = _raw_out0[_out_ind.get()];'
        self._routine_cache = {}

    def __call__(self, object a, axis=None, dtype=None, ndarray out=None,
                 bint keepdims=False):

        cdef ndarray arr

        if isinstance(a, ndarray):
            arr = a
        elif hasattr(a, '__cuda_array_interface__'):
            arr = _convert_object_with_cuda_array_interface(a)
        else:
            raise TypeError(
                'Argument \'a\' has incorrect type (expected %s, got %s)' %
                (ndarray, type(a)))
        in_args = [arr]

        dev_id = device.get_device_id()
        _check_array_device_id(arr, dev_id)

        if out is None:
            out_args = []
        else:
            _check_array_device_id(out, dev_id)
            out_args = [out]

        reduce_dims = True
        return self._call(
            in_args, out_args,
            arr.shape, axis, dtype, keepdims, reduce_dims, None)

    cdef tuple _get_expressions_and_types(
            self, list in_args, list out_args, dtype):

        in_types, out_types, routine = _guess_routine(
            self.name, self._routine_cache, self._ops, in_args, dtype)
        map_expr, reduce_expr, post_map_expr, reduce_type = routine

        if reduce_type is None:
            reduce_type = _get_typename(out_types[0])

        if out_args:
            out_type = out_args[0].dtype.type
        else:
            out_type = out_types[0]

        types = (
            ('type_in0_raw', in_args[0].dtype.type),
            ('type_out0_raw', out_type))

        return (
            map_expr, reduce_expr, post_map_expr,
            in_types, out_types, reduce_type,
            types)

    cdef list _get_out_args(
            self, list out_args, tuple out_types, tuple out_shape):
        return _get_out_args(
            out_args, out_types, out_shape, 'unsafe')

    cdef function.Function _get_function(
            self,
            tuple params, tuple args_info, tuple types,
            str map_expr, str reduce_expr, str post_map_expr, str reduce_type,
            Py_ssize_t block_size, bint is_c_contiguous):
        return _SimpleReductionKernel_get_cached_function(
            map_expr, reduce_expr, post_map_expr, reduce_type,
            params, args_info, types,
            self.name, block_size, self.identity,
            self._input_expr, self._output_expr, self._preamble, (),
            is_c_contiguous)


@util.memoize(for_each_device=True)
def _SimpleReductionKernel_get_cached_function(
        map_expr, reduce_expr, post_map_expr, reduce_type,
        params, args_info, types,
        name, block_size, identity, input_expr, output_expr, _preamble,
        options, is_c_contiguous):
    params = _get_kernel_params(params, args_info)

    type_preamble = '\n'.join(
        'typedef %s %s;' % (_get_typename(v), k)
        for k, v in types)

    return _create_reduction_function(
        name, block_size, reduce_type, params, identity,
        map_expr, reduce_expr, post_map_expr,
        type_preamble, input_expr, output_expr, _preamble, options,
        is_c_contiguous)


# -----------------------------------------------------------------------------
# ReductionKernel
# -----------------------------------------------------------------------------


cdef class ReductionKernel(_AbstractReductionKernel):

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

    cdef:
        readonly int nin
        readonly int nout
        readonly int nargs
        readonly tuple params
        readonly str reduce_expr
        readonly str map_expr
        readonly str post_map_expr
        readonly object options
        readonly bint reduce_dims
        readonly object reduce_type
        readonly str preamble

    def __init__(self, str in_params, str out_params,
                 map_expr, reduce_expr, post_map_expr,
                 identity, name='reduce_kernel', reduce_type=None,
                 reduce_dims=True, preamble='', options=()):
        if not compiler.is_valid_kernel_name(name):
            raise ValueError(
                'Invalid kernel name: "%s"' % name)

        super().__init__(
            name,
            '' if identity is None else str(identity),
            in_params,
            out_params,
        )
        self.nin = len(self.in_params)
        self.nout = len(self.out_params)
        self.nargs = self.nin + self.nout
        self.reduce_expr = reduce_expr
        self.map_expr = map_expr
        self.post_map_expr = post_map_expr
        self.options = options
        self.reduce_dims = reduce_dims
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

        return self._call(
            in_args, out_args,
            broad_shape, axis, None,
            keepdims, self.reduce_dims, stream)

    cdef tuple _get_expressions_and_types(
            self, list in_args, list out_args, dtype):

        in_ndarray_types = tuple(
            [a.dtype.type if isinstance(a, ndarray) else None
             for a in in_args])
        out_ndarray_types = tuple(
            [a.dtype.type if isinstance(a, ndarray) else None
             for a in out_args])
        in_types, out_types, types = _decide_params_type(
            self.in_params, self.out_params,
            in_ndarray_types, out_ndarray_types)
        return (
            self.map_expr, self.reduce_expr, self.post_map_expr,
            in_types, out_types, self.reduce_type,
            types)

    cdef list _get_out_args(
            self, list out_args, tuple out_types, tuple out_shape):
        return _get_out_args_with_params(
            out_args, out_types, out_shape, self.out_params, False)

    cdef function.Function _get_function(
            self,
            tuple params, tuple args_info, tuple types,
            str map_expr, str reduce_expr, str post_map_expr, str reduce_type,
            Py_ssize_t block_size, bint is_c_contiguous):
        return _ReductionKernel_get_cached_function(
            self.nin, self.nout, params, args_info, types,
            self.name, block_size, reduce_type, self.identity,
            map_expr, reduce_expr, post_map_expr,
            self.preamble, self.options, is_c_contiguous)


@util.memoize(for_each_device=True)
def _ReductionKernel_get_cached_function(
        nin, nout, params, args_info, types,
        name, block_size, reduce_type, identity, map_expr, reduce_expr,
        post_map_expr, preamble, options, is_c_contiguous):
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

    return _create_reduction_function(
        name, block_size, reduce_type, kernel_params, identity,
        map_expr, reduce_expr, post_map_expr,
        type_preamble, input_expr, output_expr, preamble, options,
        is_c_contiguous)
