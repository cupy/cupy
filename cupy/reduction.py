import collections
import string

import numpy

import cupy
from cupy import carray
from cupy import cindexer
from cupy import elementwise
from cupy import internal
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
        axis = tuple(numpy.arange(ndim, dtype=int))
    elif isinstance(axis, collections.Iterable):
        axis = tuple(axis)
    else:
        axis = axis,

    if any(ax < -ndim or ax >= ndim for ax in axis):
        raise ValueError('Axis overrun')

    axis = tuple(sorted([ax if ax >= 0 else ax + ndim for ax in axis]))
    return axis


def _get_out_shape(shape, axis, keepdims):
    if keepdims:
        out_shape = list(shape)
        for i in axis:
            out_shape[i] = 1
        out_shape = tuple(out_shape)
    else:
        out_shape = tuple(numpy.delete(shape, axis))
    return out_shape


def _get_trans_args(args, axis, ndim, params=None):
    raw_axis = numpy.arange(ndim, dtype=int)
    trans = axis + tuple(numpy.delete(raw_axis, axis))

    if all(i == j for i, j in zip(trans, raw_axis)):
        return args
    if params is not None and any(p.raw for p in params):
        raise NotImplementedError('Illegal conditions')
    return [a.transpose(trans) if isinstance(a, cupy.ndarray) else a
            for a in args]


def _get_inout_args(in_args, out_args, in_indexer, out_indexer, out_clp2_size,
                    params, reduce_dims):
    if reduce_dims:
        in_args, in_indexer = elementwise._reduce_dims(
            in_args, params, in_indexer)
        out_args, out_indexer = elementwise._reduce_dims(
            out_args, params[len(in_args):], out_indexer)
    args = in_args + out_args + [in_indexer, out_indexer,
                                 numpy.int32(out_clp2_size)]
    is_ndarray = tuple(isinstance(x, cupy.ndarray) for x in args)
    return args, is_ndarray


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

    def __call__(self, a, axis=None, dtype=None, out=None, keepdims=False,
                 allocator=None):
        if not isinstance(a, cupy.ndarray):
            raise TypeError('Input type must be cupy.ndarray')

        if self.identity is None:
            assert a.size != 0
        in_args = [a]
        if out is None:
            out_args = []
        else:
            out_args = [out]
        internal.check_args_device(in_args + out_args)

        in_types, out_types, routine = self._guess_routine(in_args, dtype)

        axis = _get_axis(axis, a.ndim)
        out_shape = _get_out_shape(a.shape, axis, keepdims)
        out_args = elementwise._get_out_args(
            in_args, out_args, out_types, allocator, out_shape)
        in_args = _get_trans_args(in_args, axis, in_args[0].ndim)

        in_indexer = cindexer.Indexer(in_args[0].shape)
        out_indexer = cindexer.Indexer(out_shape)
        out_clp2_size = 2 ** int.bit_length(int(out_indexer.size - 1))

        inout_args, is_ndarray = _get_inout_args(
            in_args, out_args, in_indexer, out_indexer, out_clp2_size,
            self._params, True)
        param_types = elementwise._get_kernel_param_types(inout_args)
        params = elementwise._get_kernel_params(
            self._params, is_ndarray, param_types)

        block_size = 512
        reduce_type = routine[3]
        if reduce_type is None:
            reduce_type = elementwise._get_typename(out_types[0])

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
            type_preamble, self._input_expr, self._output_expr, self._preamble)
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

    def _guess_routine(self, in_args, dtype):
        if dtype is None:
            for in_types, out_types, routine in self._ops:
                if all(numpy.can_cast(in_arg, in_type)
                       for in_arg, in_type in zip(in_args, in_types)):
                    return in_types, out_types, routine
        else:
            for in_types, out_types, routine in self._ops:
                if all(t == dtype for t in out_types):
                    return in_types, out_types, routine
        raise TypeError('Wrong type of arguments for %s' % self.name)


@util.memoize(for_each_device=True)
def _get_reduction_kernel(
        params, is_ndarray, param_types, types):
    kernel_params = elementwise._get_kernel_params(
        params, is_ndarray, param_types)
    type_preamble = '\n'.join(
        'typedef {} {};'.format(elementwise._get_typename(v), k)
        for k, v in types)
    input_expr = '\n'.join(
        'const {0} {1} = _raw_{1}[_j];'.format(p.ctype, p.name)
        for f, p in zip(is_ndarray, params) if f and p.const and not p.raw)
    output_expr = '\n'.join(
        '{0} &{1} = _raw_{1}[_i];'.format(p.ctype, p.name)
        for f, p in zip(is_ndarray, params) if f and not p.const and not p.raw)
    return kernel_params, type_preamble, input_expr, output_expr


class ReductionKernel(object):

    """User-defined full reduction kernel.

    This class can be used to define a PyCUDA-style full reduction kernel. It
    can accept an arbitrary number of arguments of either scalars or arrays.
    User just have to define the *map* and *reduce* operations in CUDA-C/C++.
    The map operation is defined with the special variable ``i`` that refers to
    the indices running through all the elements of the first array argument.

    The kernel is compiled at an invocation of the
    :meth:`ReductionKernel.__call__` method, which is cached for each device.
    The compiled binary is also cached into a file under the
    ``$HOME/.cupy/kernel_cache/`` directory with a hashed file name. The cached
    binary is reused by other processes.

    Args:
        param_names (list): List of argument names. Note that the type of the
            arguments are automatically determined at invocations.
        map_expr (str): The map operation definition in CUDA-C/C++. The index
            can be referred by the variable ``i``.
        reduce_expr (str): The reduce operation definition in CUDA-C/C++. The
            special variable ``a`` and ``b`` can be used for the pairwise
            reduction.
        identity (str): Initial value of the reduction in CUDA-C/C++.
        name (str): Name of the kernel function. It should be set for
            readability of the performance profiling.
        out_dtype: Data type specifier of the output.
        options (list): Options passed to the nvcc command.
        reduce_dims (bool): If False, the shapes of array arguments are
            kept within the kernel invocation. THe shapes are reduced
            (i.e., the arrays are reshaped without copy to the minimum
            ndims) by default. It may make the kernel fast by reducing the
            index calculations.
        post_map_expr (str):  Fragment of the CUDA-C/C++ code that is inserted
            below the reduction code. The reduced value can be referred by the
            special variable ``a``.
        preamble (str): Fragment of the CUDA-C/C++ code that is inserted at the
            top of the cu file.

    .. admonition:: Example

       Suppose that we want to compute the Euclidean distance between two
       arrays. It can be done as a combination of vector computations, which
       needs four kernels (subtraction, square, sum, and sqrt). We can use the
       ReductionKernel class to unify the kernels as follows::

           >>> x = cupy.array([1, 2, 3, 4, 5], dtype='f')
           >>> y = cupy.array([5, 4, 3, 2, 1], dtype='f')
           >>> kernel = cupy.reduction.ReductionKernel(
           ...     ['x', 'y'],
           ...     'squared_diff(x[i], y[i])',
           ...     'a+b',
           ...     '0',
           ...     'euclidean_distance',
           ...     post_map_expr='sqrt(a)',
           ...     preamble='''
           ...         __device__ float squared_diff(float x, float y) {
           ...             return (x - y) * (x - y);
           ...         }
           ...     ''')
           >>> z = kernel(x, y)
           >>> z
           array(6.324555397033691, dtype=float32)

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
        """Compiles and invokes the full reduction kernel.

        The compilation runs only if the kernel is not cached. Note that the
        kernels with different argument dtypes or ndims are not compatible. It
        means that single ReductionKernel object may be compiled into multiple
        kernel binaries.

        Args:
            args: Arguments of the kernel.

        Returns:
            cupy.ndarray: The result in zero-dimensional array.

        """

        out = kwargs.pop('out', None)
        axis = kwargs.get('axis', None)
        keepdims = kwargs.get('keepdims', False)
        allocator = kwargs.get('allocator', None)

        if not (len(args) == self.nin or
                len(args) == self.nin + self.nout):
            raise TypeError('Wrong number of arguments for %s' % self.name)
        assert all(i is not None for i in args)

        out_args = list(args[self.nin:])
        if out is not None:
            if self.nout != 1:
                raise NotImplementedError('')
            if len(out_args) != 0:
                raise ValueError("cannot specify 'out' as both "
                                 "a positional and keyword argument")
            out_args = [out]

        brod, in_args = elementwise._broadcast(args, self.in_params)

        internal.check_args_device(in_args + out_args)

        if self.identity is None:
            assert brod.size != 0

        in_types, out_types, types = elementwise._decide_params_type(
            self.in_params, self.out_params,
            elementwise._get_ndarray_dtype(in_args),
            elementwise._get_ndarray_dtype(out_args))

        axis = _get_axis(axis, brod.nd)
        in_args = [x if isinstance(x, cupy.ndarray) else t.type(x)
                   for x, t in zip(in_args, in_types)]
        in_args = _get_trans_args(in_args, axis, brod.nd, self.in_params)
        out_shape = _get_out_shape(brod.shape, axis, keepdims)
        out_args = elementwise._get_out_args(
            in_args, out_args, out_types, allocator, out_shape,
            self.out_params)

        in_indexer = cindexer.Indexer(brod.shape)
        out_indexer = cindexer.Indexer(out_shape)
        out_clp2_size = 2 ** int.bit_length(int(out_indexer.size - 1))

        inout_args, is_ndarray = _get_inout_args(
            in_args, out_args, in_indexer, out_indexer, out_clp2_size,
            self.params, self.reduce_dims)
        param_types = elementwise._get_kernel_param_types(inout_args)

        exprs = _get_reduction_kernel(
            self.params, is_ndarray, param_types, types)
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
            rt = tuple(i or j for i, j in zip(rt, routine))

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
