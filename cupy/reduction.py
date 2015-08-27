import collections
import string

import numpy
import six

import cupy
from cupy import carray
from cupy import elementwise
from cupy import util


@util.memoize(for_each_device=True)
def _make_reduction_function_kernel(
        name, block_size, reduce_type, params, identity,
        pre_map_expr, reduce_expr, post_map_expr,
        type_preamble, input_expr, output_expr, preamble):
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
    module = carray.compile_with_cache(module_code)
    return module.get_function(name)


def _get_axis(axis, ndim):
    if axis is None:
        axis = tuple(six.moves.range(ndim))
    elif isinstance(axis, collections.Iterable):
        axis = tuple(axis)
    else:
        axis = axis,

    if any(dim < -ndim or dim >= ndim for dim in axis):
        raise ValueError('Axis overrun')
    axis = tuple(sorted([dim % ndim for dim in axis]))
    raxis = tuple([dim for dim in six.moves.range(ndim) if dim not in axis])
    return axis, raxis


def _get_out_shape(shape, axis, raxis, keepdims):
    if keepdims:
        out_shape = list(shape)
        for i in axis:
            out_shape[i] = 1
        return tuple(out_shape)
    return tuple([shape[i] for i in raxis])


def _get_trans_args(args, trans, shape, params=None):
    if trans == tuple(six.moves.range(len(shape))):
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
        in_args, in_indexer = elementwise._reduce_dims(
            in_args, params, in_indexer)
        out_args, out_indexer = elementwise._reduce_dims(
            out_args, params[len(in_args):], out_indexer)
    args = in_args + out_args + [in_indexer, out_indexer,
                                 numpy.int32(out_clp2_size)]
    return args


class simple_reduction_function(object):

    def __init__(self, name, ops, identity=None, preamble=''):
        self.name = name
        self._ops = ops
        self.identity = identity
        self._preamble = preamble
        self.nin = 1
        self.nout = 1
        in_params = elementwise._get_param_info('T in0', True)
        out_params = elementwise._get_param_info('T out0')
        self._params = in_params + out_params + elementwise._get_param_info(
            'CIndexer _in_ind, CIndexer _out_ind, int32 _out_clp2_size')
        self._input_expr = 'const type_in0_raw in0 = _raw_in0[_in_ind.get()];'
        self._output_expr = 'type_out0_raw &out0 = _raw_out0[_out_ind.get()];'
        self._routine_cache = {}

    def __call__(self, a, axis=None, dtype=None, out=None, keepdims=False):
        if not isinstance(a, cupy.ndarray):
            raise TypeError('Input type must be cupy.ndarray')

        if self.identity is None:
            assert a.size != 0
        in_args = [a]
        if out is None:
            elementwise._check_args((a,))
            out_args = []
        else:
            elementwise._check_args((a, out))
            out_args = [out]

        in_types, out_types, routine = elementwise._guess_routine(
            self.name, self._routine_cache, self._ops, in_args, dtype)

        axis, raxis = _get_axis(axis, a.ndim)
        out_shape = _get_out_shape(a.shape, axis, raxis, keepdims)
        out_args = elementwise._get_out_args(
            out_args, out_types, out_shape)
        in_args, in_shape = _get_trans_args(
            in_args, axis + raxis, in_args[0].shape)

        in_indexer = carray.Indexer(in_shape)
        out_indexer = carray.Indexer(out_shape)
        out_clp2_size = 2 ** int.bit_length(int(out_indexer.size - 1))

        inout_args = _get_inout_args(
            in_args, out_args, in_indexer, out_indexer, out_clp2_size,
            self._params, True)
        args_info = elementwise._get_args_info(inout_args)

        block_size = 512
        reduce_type = routine[3]
        if reduce_type is None:
            reduce_type = elementwise._get_typename(out_types[0])

        params = elementwise._get_kernel_params(self._params, args_info)
        type_preamble = (
            'typedef {} type_in0_raw; typedef {} type_out0_raw;'.format(
                elementwise._get_typename(in_args[0].dtype),
                elementwise._get_typename(out_args[0].dtype)))

        kern = _make_reduction_function_kernel(
            self.name,
            block_size,
            reduce_type,
            params,
            self.identity,
            routine[0], routine[1], routine[2],
            type_preamble, self._input_expr, self._output_expr,
            self._preamble)
        shared_mem = 32 * block_size
        if out_clp2_size > 256:
            shared_mem = 0
        # TODO(okuta) set actual size
        kern.linear_launch(max(out_indexer.size, block_size), inout_args,
                           shared_mem=shared_mem,
                           block_max_size=block_size)

        if len(out_args) == 1:
            return out_args[0]
        return tuple(out_args)


@util.memoize(for_each_device=True)
def _get_reduction_kernel(params, args_info, types):
    kernel_params = elementwise._get_kernel_params(params, args_info)
    arrays = [p for p, a in six.moves.zip(params, args_info)
              if not p.raw and a[0] is cupy.ndarray]
    type_preamble = '\n'.join(
        'typedef {} {};'.format(elementwise._get_typename(v), k)
        for k, v in types)
    input_expr = '\n'.join(
        'const {0} {1} = _raw_{1}[_j];'.format(p.ctype, p.name)
        for p in arrays if p.is_const)
    output_expr = '\n'.join(
        '{0} &{1} = _raw_{1}[_i];'.format(p.ctype, p.name)
        for p in arrays if not p.is_const)
    return kernel_params, type_preamble, input_expr, output_expr


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
        options (tuple of str): Additional compilation options.
        reduce_dims (bool): If True, input arrays are reshaped without copy to
            smaller dimensions for efficiency.
        preamble (str): Fragment of the CUDA-C/C++ code that is inserted at the
            top of the cu file.

    """
    def __init__(self, in_params, out_params,
                 map_expr, reduce_expr, post_map_expr,
                 identity, name='reduce_kernel', reduce_type=None,
                 options=(), reduce_dims=True, preamble=''):
        self.in_params = elementwise._get_param_info(in_params, True)
        self.out_params = elementwise._get_param_info(out_params)
        self.nin = len(self.in_params)
        self.nout = len(self.out_params)
        self.params = self.in_params + self.out_params + \
            elementwise._get_param_info(
                'CIndexer _in_ind, CIndexer _out_ind, int32 _out_clp2_size')
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
        axis = kwargs.get('axis', None)
        keepdims = kwargs.get('keepdims', False)

        if not (len(args) == self.nin or
                len(args) == self.nin + self.nout):
            raise TypeError('Wrong number of arguments for %s' % self.name)

        out_args = list(args[self.nin:])
        if out is not None:
            if self.nout != 1:
                raise NotImplementedError('')
            if len(out_args) != 0:
                raise ValueError("cannot specify 'out' as both "
                                 "a positional and keyword argument")
            out_args = [out]

        in_args, broad_shape = elementwise._broadcast(args, self.in_params)
        elementwise._check_args(in_args + out_args)

        if self.identity is None:
            assert 0 in broad_shape
        in_types, out_types, types = elementwise._decide_params_type(
            self.in_params, self.out_params,
            elementwise._get_ndarray_dtype(in_args),
            elementwise._get_ndarray_dtype(out_args))

        axis, raxis = _get_axis(axis, len(broad_shape))
        out_shape = _get_out_shape(broad_shape, axis, raxis, keepdims)
        in_args = [x if isinstance(x, cupy.ndarray) else t.type(x)
                   for x, t in six.moves.zip(in_args, in_types)]
        in_args, in_shape = _get_trans_args(
            in_args, axis + raxis, broad_shape, self.in_params)
        out_args = elementwise._get_out_args(
            out_args, out_types, out_shape, self.out_params)

        in_indexer = carray.Indexer(in_shape)
        out_indexer = carray.Indexer(out_shape)
        out_clp2_size = 2 ** int.bit_length(int(out_indexer.size - 1))

        inout_args = _get_inout_args(
            in_args, out_args, in_indexer, out_indexer, out_clp2_size,
            self.params, self.reduce_dims)
        args_info = elementwise._get_args_info(inout_args)

        exprs = _get_reduction_kernel(
            self.params, args_info, types)
        block_size = 512
        kern = _make_reduction_function_kernel(
            self.name, block_size, self.reduce_type, exprs[0], self.identity,
            self.map_expr, self.reduce_expr, self.post_map_expr,
            exprs[1], exprs[2], exprs[3], self.preamble)
        shared_mem = 32 * block_size
        if out_clp2_size > 256:
            shared_mem = 0
        # TODO(okuta) set actual size
        kern.linear_launch(max(out_indexer.size, block_size), inout_args,
                           shared_mem=shared_mem,
                           block_max_size=block_size)
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
            rt = tuple(i or j for i, j in six.moves.zip(rt, routine))

        types = typ.split('->')
        if len(types) == 1:
            in_types = out_types = tuple(types)
        else:
            in_types, out_types = map(tuple, types)
        in_types = tuple(numpy.dtype(t) for t in in_types)
        out_types = tuple(numpy.dtype(t) for t in out_types)
        _ops.append((in_types, out_types, rt))

    return simple_reduction_function(
        name, _ops, identity=identity, preamble=preamble)


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
    ['?->?', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'],
    ('min_max_st(in0)', 'my_min(a, b)', 'out0 = a.value', 'min_max_st'),
    None, _min_max_preamble)

amax = create_reduction_func(
    'cupy_max',
    ['?->?', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'],
    ('min_max_st(in0)', 'my_max(a, b)', 'out0 = a.value', 'min_max_st'),
    None, _min_max_preamble)

argmin = create_reduction_func(
    'cupy_argmin',
    ['?->l', 'B->l', 'h->l', 'H->l', 'i->l', 'I->l', 'l->l', 'L->l',
     'q->l', 'Q->l', 'e->l', 'f->l', 'd->l'],
    ('min_max_st(in0, _J)', 'my_argmin(a, b)', 'out0 = a.index', 'min_max_st'),
    None, _min_max_preamble)

argmax = create_reduction_func(
    'cupy_argmax',
    ['?->l', 'B->l', 'h->l', 'H->l', 'i->l', 'I->l', 'l->l', 'L->l',
     'q->l', 'Q->l', 'e->l', 'f->l', 'd->l'],
    ('min_max_st(in0, _J)', 'my_argmax(a, b)', 'out0 = a.index', 'min_max_st'),
    None, _min_max_preamble)
