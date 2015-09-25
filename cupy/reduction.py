import collections
import string

import numpy
import six

import cupy
from cupy import carray
from cupy import elementwise
from cupy import util


six_range = six.moves.range
six_zip = six.moves.zip

_broadcast = elementwise._broadcast
_check_args = elementwise._check_args
_decide_params_type = elementwise._decide_params_type
_get_kernel_params = elementwise._get_kernel_params
_get_args_info = elementwise._get_args_info
_get_out_args = elementwise._get_out_args
_get_out_args_with_params = elementwise._get_out_args_with_params
_get_param_info = elementwise._get_param_info
_get_typename = elementwise._get_typename
_guess_routine = elementwise._guess_routine
_reduce_dims = elementwise._reduce_dims


def _get_simple_reduction_kernel(
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

    typedef ${reduce_type} _type_reduce;
    extern "C" __global__ void ${name}(${params}) {
      if (_out_clp2_size > 256) {
        CUPY_FOR(_i, _out_ind.size()) {
          _type_reduce _s = _type_reduce(${identity});
          for (int _j = _i, _J = 0;
               _j < _in_ind.size();
               _j += _out_ind.size(), _J++) {
            _in_ind.set(_j);
            ${input_expr}
            _type_reduce _a = ${pre_map_expr};
            _s = REDUCE(_s, _a);
          }
          _out_ind.set(_i);
          ${output_expr}
          POST_MAP(_s);
        }
      } else {
        extern __shared__ _type_reduce _sdata_raw[];
        _type_reduce *_sdata = _sdata_raw;
        int _tid = threadIdx.x;
        _sdata[_tid] = _type_reduce(${identity});
        unsigned int _i = _tid % _out_clp2_size;
        if (_i >= _out_ind.size()) return;
        _type_reduce _s = _type_reduce(${identity});
        int _J_offset = _tid / _out_clp2_size;
        int _j_offset = _J_offset * _out_ind.size();
        int _J_stride = ${block_size} / _out_clp2_size;
        int _j_stride = _J_stride * _out_ind.size();
        for (int _j = _i + _j_offset, _J = _J_offset;
             _j < _in_ind.size();
             _j += _j_stride, _J += _J_stride) {
          _in_ind.set(_j);
          ${input_expr}
          _type_reduce _a = ${pre_map_expr};
          _s = REDUCE(_s, _a);
        }
        _sdata[_tid] = _s;
        __syncthreads();
        if (_tid >= 256) return;
        _sdata[_tid] = REDUCE(_sdata[_tid], _sdata[_tid + 256]);
        __syncthreads();
        if (_out_clp2_size <= 128) {
          _sdata[_tid] = REDUCE(_sdata[_tid], _sdata[_tid + 128]);
          __syncthreads();
          if (_out_clp2_size <= 64) {
            _sdata[_tid] = REDUCE(_sdata[_tid], _sdata[_tid + 64]);
            __syncthreads();
            if (_out_clp2_size <= 32) {
              _sdata[_tid] = REDUCE(_sdata[_tid], _sdata[_tid + 32]);
              if (_out_clp2_size <= 16) {
                _sdata[_tid] = REDUCE(_sdata[_tid], _sdata[_tid + 16]);
                if (_out_clp2_size <= 8) {
                  _sdata[_tid] = REDUCE(_sdata[_tid], _sdata[_tid + 8]);
                  if (_out_clp2_size <= 4) {
                    _sdata[_tid] = REDUCE(_sdata[_tid], _sdata[_tid + 4]);
                    if (_out_clp2_size <= 2) {
                      _sdata[_tid] = REDUCE(_sdata[_tid], _sdata[_tid + 2]);
                      if (_out_clp2_size <= 1) {
                        _sdata[_tid] = REDUCE(_sdata[_tid], _sdata[_tid + 1]);
                      }
                    }
                  }
                }
              }
            }
          }
        }
        _s = _sdata[_tid];
        if (_tid >= _out_ind.size()) return;
        _out_ind.set(_i);
         ${output_expr}
        POST_MAP(_s);
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
    module = carray.compile_with_cache(module_code, options)
    return module.get_function(name)


def _get_axis(axis, ndim):
    if axis is None:
        axis = tuple(six_range(ndim))
    elif isinstance(axis, collections.Sequence):
        axis = tuple(axis)
    else:
        axis = axis,

    for dim in axis:
        if dim < -ndim or dim >= ndim:
            raise ValueError('Axis overrun')
    axis = tuple(sorted([dim % ndim for dim in axis]))
    raxis = tuple([dim for dim in six_range(ndim) if dim not in axis])
    return axis, raxis


def _get_out_shape(shape, axis, raxis, keepdims):
    if keepdims:
        out_shape = list(shape)
        for i in axis:
            out_shape[i] = 1
        return tuple(out_shape)
    return tuple([shape[i] for i in raxis])


def _get_trans_args(args, trans, shape, params=None):
    if trans == tuple(six_range(len(shape))):
        return args, shape
    if params is not None and any(p.raw for p in params):
        raise NotImplementedError('Illegal conditions')
    args = [cupy.transpose(a, trans) if isinstance(a, cupy.ndarray) else a
            for a in args]
    shape = tuple([shape[i] for i in trans])
    return args, shape


def _get_inout_args(in_args, out_args, in_indexer, out_indexer, out_clp2_size,
                    params, reduce_dims):
    if reduce_dims:
        in_args, in_shape = _reduce_dims(
            in_args, params, in_indexer.shape)
        out_args, out_shape = _reduce_dims(
            out_args, params[len(in_args):], out_indexer.shape)
        in_indexer.shape = in_shape
        out_indexer.shape = out_shape
    args = in_args + out_args + [in_indexer, out_indexer,
                                 numpy.int32(out_clp2_size)]
    return args


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
            _get_param_info(
                'CIndexer _in_ind, CIndexer _out_ind', False) +
            _get_param_info('int32 _out_clp2_size', True))
        self._input_expr = 'const type_in0_raw in0 = _raw_in0[_in_ind.get()];'
        self._output_expr = 'type_out0_raw &out0 = _raw_out0[_out_ind.get()];'
        self._routine_cache = {}

    def __call__(self, a, axis=None, dtype=None, out=None, keepdims=False):
        if not isinstance(a, cupy.ndarray):
            raise TypeError('Input type must be cupy.ndarray')
        if self.identity is None:
            assert a.size != 0
        if dtype is not None:
            dtype = numpy.dtype(dtype).type

        in_args = [a]
        if out is None:
            _check_args((a,))
            out_args = []
        else:
            _check_args((a, out))
            out_args = [out]

        in_types, out_types, routine = _guess_routine(
            self.name, self._routine_cache, self._ops, in_args, dtype)

        axis, raxis = _get_axis(axis, a.ndim)
        out_shape = _get_out_shape(a.shape, axis, raxis, keepdims)
        out_args = _get_out_args(out_args, out_types, out_shape)
        in_args, in_shape = _get_trans_args(
            in_args, axis + raxis, in_args[0].shape)

        in_indexer = carray.Indexer(in_shape)
        out_indexer = carray.Indexer(out_shape)
        out_clp2_size = 2 ** int.bit_length(int(out_indexer.size - 1))

        inout_args = _get_inout_args(
            in_args, out_args, in_indexer, out_indexer, out_clp2_size,
            self._params, True)
        args_info = _get_args_info(inout_args)

        block_size = 512
        kern = _get_simple_reduction_function(
            routine, self._params, args_info,
            in_args[0].dtype.type, out_args[0].dtype.type, out_types,
            self.name, block_size, self.identity,
            self._input_expr, self._output_expr, self._preamble, ())

        shared_mem = 32 * block_size
        if out_clp2_size > 256:
            shared_mem = 0
        # TODO(okuta) set actual size
        kern.linear_launch(max(out_indexer.size, block_size), inout_args,
                           shared_mem, block_size)

        if len(out_args) == 1:
            return out_args[0]
        return tuple(out_args)


@util.memoize(for_each_device=True)
def _get_reduction_kernel(
        params, args_info, types,
        name, block_size, reduce_type, identity, map_expr, reduce_expr,
        post_map_expr, preamble, options):
    kernel_params = _get_kernel_params(params, args_info)
    arrays = [p for p, a in six_zip(params, args_info)
              if not p.raw and a[0] is cupy.ndarray]
    type_preamble = '\n'.join(
        'typedef %s %s;' % (_get_typename(v), k)
        for k, v in types)
    input_expr = '\n'.join(
        ['const {0} {1} = _raw_{1}[_j];'.format(p.ctype, p.name)
         for p in arrays if p.is_const])
    output_expr = '\n'.join(
        ['{0} &{1} = _raw_{1}[_i];'.format(p.ctype, p.name)
         for p in arrays if not p.is_const])

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
    binary is resued by other processes.

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
        reduce_dims (bool): If True, input arrays are reshaped without copy to
            smaller dimensions for efficiency.
        preamble (str): Fragment of the CUDA-C/C++ code that is inserted at the
            top of the cu file.
        options (tuple of str): Additional compilation options.

    """
    def __init__(self, in_params, out_params,
                 map_expr, reduce_expr, post_map_expr,
                 identity, name='reduce_kernel', reduce_type=None,
                 reduce_dims=True, preamble='', options=()):
        self.in_params = _get_param_info(in_params, True)
        self.out_params = _get_param_info(out_params, False)
        self.nin = len(self.in_params)
        self.nout = len(self.out_params)
        self.nargs = self.nin + self.nout
        self.params = (
            self.in_params + self.out_params +
            _get_param_info('CIndexer _in_ind, CIndexer _out_ind', False) +
            _get_param_info('int32 _out_clp2_size', True))
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

        Returns:
            Arrays are returned according to the ``out_params`` argument of the
            ``__init__`` method.

        """

        out = kwargs.pop('out', None)
        axis = kwargs.pop('axis', None)
        keepdims = kwargs.pop('keepdims', False)
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

        in_args, broad_shape = _broadcast(args, self.in_params, False)
        _check_args(in_args + out_args)

        if self.identity is None:
            assert 0 in broad_shape

        cp_array = cupy.ndarray
        in_ndarray_types = tuple(
            [a.dtype.type if isinstance(a, cp_array) else None
             for a in in_args])
        out_ndarray_types = tuple(
            [a.dtype.type if isinstance(a, cp_array) else None
             for a in out_args])
        in_types, out_types, types = _decide_params_type(
            self.in_params, self.out_params,
            in_ndarray_types, out_ndarray_types)

        axis, raxis = _get_axis(axis, len(broad_shape))
        out_shape = _get_out_shape(broad_shape, axis, raxis, keepdims)
        in_args = [x if isinstance(x, cp_array) else t(x)
                   for x, t in six_zip(in_args, in_types)]
        in_args, in_shape = _get_trans_args(
            in_args, axis + raxis, broad_shape, self.in_params)
        out_args = _get_out_args_with_params(
            out_args, out_types, out_shape, self.out_params)

        in_indexer = carray.Indexer(in_shape)
        out_indexer = carray.Indexer(out_shape)
        out_clp2_size = 2 ** int.bit_length(int(out_indexer.size - 1))

        inout_args = _get_inout_args(
            in_args, out_args, in_indexer, out_indexer, out_clp2_size,
            self.params, self.reduce_dims)
        args_info = _get_args_info(inout_args)

        block_size = 512
        kern = _get_reduction_kernel(
            self.params, args_info, types,
            self.name, block_size, self.reduce_type, self.identity,
            self.map_expr, self.reduce_expr, self.post_map_expr,
            self.preamble, self.options)

        shared_mem = 32 * block_size
        if out_clp2_size > 256:
            shared_mem = 0
        # TODO(okuta) set actual size
        kern.linear_launch(max(out_indexer.size, block_size), inout_args,
                           shared_mem, block_size)
        return out_args[0]


def create_reduction_func(name, ops, routine=None, identity=None,
                          preamble=''):
    _ops = []
    for t in ops:
        if not isinstance(t, tuple):
            typ = t
            rt = routine
        else:
            typ, rt = t
            rt = tuple(i or j for i, j in six_zip(rt, routine))

        types = typ.split('->')
        if len(types) == 1:
            in_types = out_types = tuple(types)
        else:
            in_types, out_types = map(tuple, types)
        in_types = tuple([numpy.dtype(t).type for t in in_types])
        out_types = tuple([numpy.dtype(t).type for t in out_types])
        _ops.append((in_types, out_types, rt))

    return simple_reduction_function(name, _ops, identity, preamble)


_min_max_preamble = '''
struct min_max_st{
    type_in0_raw value;
    int index;
    __device__ min_max_st() : index(-1) { }
    __device__ min_max_st(type_in0_raw v) : value(v), index(0) { }
    __device__ min_max_st(type_in0_raw v, int i) : value(v), index(i) { }
};
__device__ min_max_st my_min(const min_max_st& a, const min_max_st& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return min_max_st(min(a.value, b.value));
}
__device__ min_max_st my_max(const min_max_st& a, const min_max_st& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return min_max_st(max(a.value, b.value));
}
__device__ min_max_st my_argmin(const min_max_st& a, const min_max_st& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return (a.value <= b.value) ? a : b;
}
__device__ min_max_st my_argmax(const min_max_st& a, const min_max_st& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return (a.value >= b.value) ? a : b;
}'''


amin = create_reduction_func(
    'cupy_min',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'),
    ('min_max_st(in0)', 'my_min(a, b)', 'out0 = a.value', 'min_max_st'),
    None, _min_max_preamble)

amax = create_reduction_func(
    'cupy_max',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'),
    ('min_max_st(in0)', 'my_max(a, b)', 'out0 = a.value', 'min_max_st'),
    None, _min_max_preamble)

argmin = create_reduction_func(
    'cupy_argmin',
    ('?->l', 'B->l', 'h->l', 'H->l', 'i->l', 'I->l', 'l->l', 'L->l',
     'q->l', 'Q->l', 'e->l', 'f->l', 'd->l'),
    ('min_max_st(in0, _J)', 'my_argmin(a, b)', 'out0 = a.index', 'min_max_st'),
    None, _min_max_preamble)

argmax = create_reduction_func(
    'cupy_argmax',
    ('?->l', 'B->l', 'h->l', 'H->l', 'i->l', 'I->l', 'l->l', 'L->l',
     'q->l', 'Q->l', 'e->l', 'f->l', 'd->l'),
    ('min_max_st(in0, _J)', 'my_argmax(a, b)', 'out0 = a.index', 'min_max_st'),
    None, _min_max_preamble)
