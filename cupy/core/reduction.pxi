import collections
import string

import numpy

from cupy.cuda import compiler
from cupy import util


cpdef tuple _get_axis(object axis, Py_ssize_t ndim):
    cdef Py_ssize_t dim
    if axis is None:
        axis = tuple(range(ndim))
    elif isinstance(axis, collections.Sequence):
        axis = tuple(axis)
    else:
        axis = axis,

    for dim in axis:
        if dim < -ndim or dim >= ndim:
            raise ValueError('Axis overrun')
    axis = tuple(sorted([dim % ndim for dim in axis]))
    raxis = tuple([dim for dim in range(ndim) if dim not in axis])
    return axis, raxis


cpdef tuple _get_out_shape(
        tuple shape, tuple axis, tuple raxis, bint keepdims):
    cdef Py_ssize_t i
    if keepdims:
        out_shape = list(shape)
        for i in axis:
            out_shape[i] = 1
        return tuple(out_shape)
    return tuple([shape[i] for i in raxis])


cpdef tuple _transpose_args(list args, list arg_infos, tuple trans, tuple shape, tuple params):
    """Transposes args in-place."""
    cdef ParameterInfo p
    cdef ArgInfo ai
    cdef Py_ssize_t i
    if trans == tuple(range(len(shape))):
        return shape
    if params is not None:
        for p in params:
            if p.raw:
                raise NotImplementedError('Illegal conditions')
    for i, ai in enumerate(arg_infos):
        if ai.is_ndarray:
            args[i] = args[i].transpose(trans)
            arg_infos[i] = ArgInfo_from_arg(args[i])
    shape = tuple([shape[i] for i in trans])
    return shape


cpdef tuple _get_inout_args(
        list in_args, list out_args, list in_arg_infos, list out_arg_infos,
        Indexer in_indexer, Indexer out_indexer,
        object out_clp2_size, tuple params, bint reduce_dims):
    if reduce_dims:
        in_shape = _reduce_dims(
            in_args, in_arg_infos, params, in_indexer.shape)
        out_shape = _reduce_dims(
            out_args, out_arg_infos, params[len(in_args):], out_indexer.shape)
        in_indexer.shape = in_shape
        out_indexer.shape = out_shape
    extra_args = [in_indexer, out_indexer, numpy.int32(out_clp2_size)]
    args = in_args + out_args + extra_args
    arg_infos = in_arg_infos + out_arg_infos + ArgInfo_from_args(extra_args)
    return args, arg_infos


cdef class _BaseReductionKernel(_BaseKernel):

    cdef:
        readonly str identity
        readonly str preamble
        readonly tuple params
        readonly bint reduce_dims
        readonly int block_size

    def __init__(
            self, in_params, out_params, str name, bint reduce_dims,
            str identity, str preamble, int block_size,
            tuple options):

        super(_BaseReductionKernel, self).__init__(
            in_params, out_params, name, options)

        self.params = (
            in_params + out_params +
            _parse_param_infos(
                'CIndexer _in_ind, CIndexer _out_ind', False) +
            _parse_param_infos('int32 _block_stride', True))
        self.reduce_dims = reduce_dims
        self.identity = identity
        self.preamble = preamble
        self.block_size = block_size


cdef class _BaseReductionKernelCallContext(_BaseKernelCallContext):
    cdef:
        readonly tuple arg_infos
        readonly str casting
        readonly object axis
        readonly bint keepdims

    def __init__(
            self, _BaseReductionKernel kernel, tuple arg_infos,
            object axis, bint keepdims, str casting):

        super(_BaseReductionKernelCallContext, self).__init__(kernel)

        self.arg_infos = arg_infos
        self.axis = axis
        self.keepdims = keepdims
        self.casting = casting

    cpdef call(self, args):
        cdef _broadcast_impl brod_impl
        cdef ArgInfo ai
        cdef Py_ssize_t i
        axis = self.axis
        keepdims = self.keepdims
        out = self.out
        casting = self.casting
        kernel = self.kernel
        nin = kernel.nin
        nout = kernel.nout
        nargs = kernel.nargs
        params = kernel.params

        args = list(args)
        in_args = args[:nin]
        out_args = args[nin:]

        # preprocess_args
        in_arg_infos = ArgInfo_from_args(in_args, True)
        out_arg_infos = ArgInfo_from_args(out_args, True)
        _preprocess_args(in_args, in_arg_infos)
        _preprocess_args(out_args, out_arg_infos)

        # Decide parameter dtypes.
        in_types, out_types = self.decide_param_types(
            in_arg_infos,
            out_arg_infos)

        # broadcast
        brod_impl, in_shape, in_arg_infos = self.broadcast_and_cast(
            in_arg_infos, -1, True)
        in_args = brod_impl.apply(in_args)

        # get out shape
        laxis, raxis = _get_axis(axis, len(in_shape))
        del axis  # to avoid bug
        out_shape = _get_out_shape(in_shape, laxis, raxis, keepdims)

        # get out args
        if len(out_args) == 0:
            out_args = _allocate_out_args(
                tuple(out_types), out_shape, None, False)
            out_arg_infos = ArgInfo_from_args(out_args)
        else:
            _check_out_args(
                out_arg_infos, tuple(out_types), out_shape, None, casting)

        if 0 in out_shape:
            if len(out_args) == 1:
                return out_args[0]
            return tuple(out_args)

        if len(out_args) == 1:
            ret = out_args[0]
        else:
            ret = tuple(out_args)

        if len(kernel.identity) == 0 and 0 in in_shape:
            raise ValueError(('zero-size array to reduction operation'
                              ' %s which has no identity') % kernel.name)

        # Cast scalar in_args to ndarrays
        for i, ai in enumerate(in_arg_infos):
            if not ai.is_ndarray:
                in_args[i] = in_types[i](in_args[i])
                in_arg_infos[i] = ArgInfo_from_arg(in_args[i])

        # Transpose in_args
        in_shape = _transpose_args(
            in_args, in_arg_infos, laxis + raxis, in_shape, None)

        block_size = kernel.block_size
        in_indexer = Indexer(in_shape)
        out_indexer = Indexer(out_shape)
        # Rounding Up to the Next Power of 2
        # clp2_count >= in_indexer.size // out_indexer.size
        clp2_count = 1 << int.bit_length(
            int(in_indexer.size // out_indexer.size - 1))
        block_stride = max(1, block_size // clp2_count)

        inout_args, inout_arg_infos = _get_inout_args(
            in_args, out_args, in_arg_infos, out_arg_infos,
            in_indexer, out_indexer, block_stride,
            params, kernel.reduce_dims)
        inout_arg_infos = tuple(inout_arg_infos)

        key = ('kernel', device.get_device_id(), inout_arg_infos) + (
            self.get_code_args())
        kern = kernel.kernel_cache.get(key)
        if kern is None:
            param_list = ParameterList(params, inout_arg_infos, nin, nout)
            params_decl = param_list.get_entry_function_params_decl()
            name = kernel.name
            loop_class_name = name + '__loop__'
            emitter = emit.KernelCodeEmitter(kernel.name)

            # Emit reduction loop computation class
            self.emit_loop_class(
                emitter, loop_class_name, param_list)

            # Emit kernel entry function
            self.emit_kernel_entry_function(
                emitter, loop_class_name, param_list)

            kern = emitter.get_function(kernel.options)
            kernel.kernel_cache[key] = kern

        # TODO(okuta) set actual size
        shared_mem = 32 * block_size

        kern.linear_launch(
            (out_indexer.size + block_stride - 1) // block_stride * block_size,
            inout_args, shared_mem, block_size)

        return ret

    def emit_loop_class(
            self, emitter, class_name, ParameterList param_list):

        kernel = <_BaseReductionKernel>self.kernel
        identity = kernel.identity
        preamble = kernel.preamble
        block_size = kernel.block_size

        # Retrieve code
        (reduce_type, pre_map_expr, reduce_expr, post_map_expr, type_preamble,
         input_expr, output_expr) = self.get_code(
             param_list, self.get_code_args())

        temp = emit.Templater(
            params_decl=param_list.get_kernel_params_decl(),
            class_name=class_name,
            preamble=emitter.indent_lines(preamble, 2),
            block_size=block_size,
            reduce_type=reduce_type,
            identity=identity,
            reduce_expr=reduce_expr,
            pre_map_expr=emitter.indent_lines(pre_map_expr, 8),
            post_map_expr=emitter.indent_lines(post_map_expr, 10),
            type_preamble=emitter.indent_lines(type_preamble, 2),
            input_expr=emitter.indent_lines(input_expr, 8),
            output_expr=output_expr)

        emitter.emit_lines(temp('''
class ${class_name} {
private:
${type_preamble}
${preamble}

  typedef ${reduce_type} _type_reduce;

  __device__ _type_reduce REDUCE(const _type_reduce& a,
                                 const _type_reduce& b) {
    return (${reduce_expr});
  }

  __device__ void _REDUCE(_type_reduce* _sdata,
                          unsigned int tid,
                          unsigned int offset) {
    if (tid < offset) {
      _type_reduce _a = _sdata[tid], _b = _sdata[(tid + offset)];
      _sdata[tid] = REDUCE(_a, _b);
    }
  }

public:
  __device__ void compute(${params_decl}) {
    extern __shared__ _type_reduce _sdata_raw[];
    _type_reduce *_sdata = _sdata_raw;
    unsigned int _tid = threadIdx.x;

    int _J_offset = _tid / _block_stride;
    int _j_offset = _J_offset * _out_ind.size();
    int _J_stride = ${block_size};
    long long _j_stride = ${block_size}LL * _out_ind.size();

    for (int _i_base = blockIdx.x * _block_stride;
         _i_base < _out_ind.size();
         _i_base += gridDim.x * _block_stride) {
      _type_reduce _s = _type_reduce(${identity});
      int _i = _i_base + _tid % _block_stride;
      int _J = _J_offset;
      for (long long _j = _i + _j_offset; _j < _in_ind.size();
           _j += _j_stride, _J += _J_stride) {
        _in_ind.set(_j);
${input_expr}
        _type_reduce _a = ${pre_map_expr};
        _s = REDUCE(_s, _a);
      }
      if (_block_stride < ${block_size}) {
        _sdata[_tid] = _s;
        __syncthreads();
        if (_block_stride <= 256) {
          _REDUCE(_sdata, _tid, 256);
          __syncthreads();
          if (_block_stride <= 128) {
            _REDUCE(_sdata, _tid, 128);
            __syncthreads();
            if (_block_stride <= 64) {
              _REDUCE(_sdata, _tid, 64);
              __syncthreads();
              if (_block_stride <= 32) {
                _REDUCE(_sdata, _tid, 32);
                if (_block_stride <= 16) {
                  _REDUCE(_sdata, _tid, 16);
                  if (_block_stride <= 8) {
                    _REDUCE(_sdata, _tid, 8);
                    if (_block_stride <= 4) {
                      _REDUCE(_sdata, _tid, 4);
                      if (_block_stride <= 2) {
                        _REDUCE(_sdata, _tid, 2);
                        if (_block_stride <= 1) {
                          _REDUCE(_sdata, _tid, 1);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        _s = _sdata[_tid];
        __syncthreads();
      }
      if (_J_offset == 0 && _i < _out_ind.size()) {
        _out_ind.set(_i);
${output_expr};
        {
          _type_reduce a = _s;  // referred in 'post_map_expr'
${post_map_expr};
        }
      }
    }
  }
};
'''))

    def emit_kernel_entry_function(
            self, emitter, str loop_class_name, ParameterList param_list):

        # Emit kernel entry function
        with emitter.construct_kernel_entry_function(param_list):
            temp = emit.Templater(
                loop_class_name=loop_class_name,
                param_list_expr=param_list.get_entry_function_param_list())
            emitter.emit_line(temp(
                '''${loop_class_name}().compute(${param_list_expr});'''))


cdef class _SimpleReductionKernelCallContext(_BaseReductionKernelCallContext):
    cdef:
        object _routine

        readonly object dtype
        readonly object out

    def __init__(
            self, simple_reduction_function kernel, tuple arg_infos,
            object axis, dtype, out, bint keepdims):

        super(_SimpleReductionKernelCallContext, self).__init__(
            kernel, arg_infos, axis, keepdims, 'unsafe')

        if dtype is not None:
            dtype = numpy.dtype(dtype)

        self.dtype = dtype
        self.out = out

    cpdef decide_param_types(self, list in_arg_infos, list out_arg_infos):
        kernel = self.kernel
        # guess_routine (and types)
        in_types, out_types, routine = _guess_routine(
            kernel.name,
            kernel._routine_cache,
            tuple(kernel._ops), in_arg_infos, self.dtype)
        self._routine = routine
        return in_types, out_types

    cpdef tuple get_code_args(self):
        return self._routine

    cpdef get_code(self, ParameterList param_list, tuple code_args):
        pre_map_expr, reduce_expr, post_map_expr, reduce_type = code_args
        if reduce_type is None:
            reduce_type = _get_ctype_name(param_list.arg_infos[1].dtype)

        type_preamble = (
            'typedef %s type_in0_raw; typedef %s type_out0_raw;' % (
                _get_ctype_name(param_list.arg_infos[0].dtype),
                _get_ctype_name(param_list.arg_infos[1].dtype)))
        input_expr = 'const type_in0_raw in0 = _raw_in0[_in_ind.get()];'
        output_expr = 'type_out0_raw &out0 = _raw_out0[_out_ind.get()];'

        return (
            reduce_type,
            pre_map_expr,
            reduce_expr,
            post_map_expr,
            type_preamble,
            input_expr,
            output_expr)


cdef class _ReductionKernelCallContext(_BaseReductionKernelCallContext):
    cdef:
        object _types
        readonly object out
        readonly object stream

    def __init__(
            self, ReductionKernel kernel, tuple arg_infos,
            object axis, bint keepdims, object out, object stream):

        super(_ReductionKernelCallContext, self).__init__(
            kernel, arg_infos, axis, keepdims, None)

        self.out = out
        self.stream = stream

    cpdef decide_param_types(self, list in_arg_infos, list out_arg_infos):
        cdef ArgInfo a
        kernel = self.kernel
        # decide param types
        in_ndarray_types = tuple(
            [a.dtype.type if a.is_ndarray else None
             for a in in_arg_infos])
        out_ndarray_types = tuple(
            [a.dtype.type if a.is_ndarray else None
             for a in out_arg_infos])
        in_types, out_types, types = _decide_param_types(
            kernel.in_params, kernel.out_params,
            in_ndarray_types, out_ndarray_types)
        self._types = types
        return in_types, out_types

    cpdef tuple get_code_args(self):
        return (self._types,)

    cpdef get_code(self, ParameterList param_list, tuple code_args):
        cdef tuple types
        cdef ParameterInfo p
        cdef ArgInfo a
        kernel = self.kernel
        types, = code_args
        array_params = [
            p for p, a in zip(param_list.params, param_list.arg_infos)
            if not p.raw and a.is_ndarray]
        type_preamble = '\n'.join(
            ['typedef %s %s;' % (_get_ctype_name(v), k)
             for k, v in types])
        input_expr = '\n'.join(
            ['const {0} {1} = _raw_{1}[_j];'.format(p.ctype, p.name)
             for p in array_params if p.is_const])
        output_expr = '\n'.join(
            ['{0} &{1} = _raw_{1}[_i];'.format(p.ctype, p.name)
             for p in array_params if not p.is_const])

        return (
            kernel.reduce_type,
            kernel.map_expr,
            kernel.reduce_expr,
            kernel.post_map_expr,
            type_preamble,
            input_expr,
            output_expr)


cdef class simple_reduction_function(_BaseReductionKernel):

    _block_size = 512

    cdef:
        readonly tuple _ops
        readonly dict _routine_cache

        object _routine

    def __init__(self, str name, tuple ops, identity, str preamble):

        if identity is None:
            identity = ''
        else:
            identity = str(identity)
        in_params = _parse_param_infos('T in0', True)
        out_params = _parse_param_infos('T out0', False)
        super(simple_reduction_function, self).__init__(
            in_params, out_params, name, True, identity, preamble,
            self._block_size, ())

        self._ops = ops
        self._routine_cache = {}

    def __call__(self, ndarray a, axis=None, dtype=None, ndarray out=None,
                 bint keepdims=False):
        if not isinstance(a, ndarray):
            raise TypeError('Input type must be cupy.ndarray')

        cdef ArgInfo out_info

        if out is not None:
            args = (a, out)
            out_info = ArgInfo_from_arg(out)
        else:
            args = (a,)
            out_info = None

        arg_infos = (ArgInfo_from_arg(a),)

        call_ctx = self.create_call_context(
            arg_infos, axis=axis, dtype=dtype, out=out_info, keepdims=keepdims)
        return call_ctx.call(args)

    cpdef create_call_context(
            self, arg_infos, axis=None, dtype=None, ArgInfo out=None,
            bint keepdims=False):

        nargs = len(arg_infos)
        if nargs != self.nin and nargs != self.nargs:
            raise TypeError('Wrong number of arguments for %s' % self.name)

        if out is not None:
            if self.nout != 1:
                raise NotImplementedError('')
            if self.nin != nargs:
                raise ValueError("cannot specify 'out' as both "
                                 "a positional and keyword argument")

        return _SimpleReductionKernelCallContext(
            self, arg_infos, axis, dtype, out, keepdims)


cdef class ReductionKernel(_BaseReductionKernel):

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
        options (tuple, list of str): Additional compilation options.

    """

    cdef:
        readonly str reduce_expr
        readonly str map_expr
        readonly str post_map_expr
        readonly str reduce_type

        object _types

    def __init__(self, str in_params, str out_params,
                 str map_expr, str reduce_expr, str post_map_expr,
                 object identity, str name='reduce_kernel',
                 str reduce_type=None, bint reduce_dims=True, str preamble='',
                 options=()):

        if isinstance(options, str):
            options = (options,)
        elif isinstance(options, list):
            options = tuple(options)

        super(ReductionKernel, self).__init__(
            _parse_param_infos(in_params, True),
            _parse_param_infos(out_params, False),
            name, reduce_dims, str(identity), preamble, 512, options)
        self.reduce_expr = reduce_expr
        self.map_expr = map_expr
        self.name = name
        self.post_map_expr = post_map_expr
        if reduce_type is None:
            self.reduce_type = self.out_params[0].ctype
        else:
            self.reduce_type = reduce_type

    def __call__(self, *args, axis=None, out=None, keepdims=None, **kwargs):
        """__call__(self, *args, axis=None, out=None, keepdims=None, **kwargs)

        Compiles and invokes the reduction kernel.

        The compilation runs only if the kernel is not cached. Note that the
        kernels with different argument dtypes, ndims, or axis are not
        compatible. It means that single ReductionKernel object may be compiled
        into multiple kernel binaries.

        Args:
            args: Arguments of the kernel.

        Returns:
            Arrays are returned according to the ``out_params`` argument of the
            ``__init__`` method.

        """

        arg_infos = tuple(ArgInfo_from_args(args))

        if out is not None:
            args += (out,)
            out = ArgInfo_from_arg(out)

        call_ctx = self.create_call_context(
            arg_infos, axis=axis, out=out, keepdims=keepdims, **kwargs)

        return call_ctx.call(args)

    cpdef create_call_context(
            self, arg_infos, axis=None, dtype=None, ArgInfo out=None,
            bint keepdims=False, stream=None):

        nargs = len(arg_infos)
        if nargs != self.nin and nargs != self.nargs:
            raise TypeError('Wrong number of arguments for %s' % self.name)

        if out is not None:
            if self.nout != 1:
                raise NotImplementedError('')
            if self.nin != nargs:
                raise ValueError("cannot specify 'out' as both "
                                 "a positional and keyword argument")

        assert out is None or isinstance(out, ArgInfo), type(out)

        return _ReductionKernelCallContext(
            self, arg_infos, axis, keepdims, out, stream)


cpdef create_reduction_func(name, ops, routine=None, identity=None,
                            preamble=''):
    if identity is None:
        identity = ''
    else:
        identity = str(identity)

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
        in_types = tuple([numpy.dtype(t).type for t in in_types])
        out_types = tuple([numpy.dtype(t).type for t in out_types])
        _ops.append((in_types, out_types, rt))

    return simple_reduction_function(name, tuple(_ops), identity, preamble)
