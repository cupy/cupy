import collections
import string

import numpy

import cupy
from cupy import carray
from cupy import cuda
from cupy import elementwise
from cupy import internal


@cuda.memoize
def _make_reduction_function_kernel(name, block_size, reduce_type, params,
                                    identity, pre_map_expr,
                                    reduce_expr,
                                    post_map_expr='a',
                                    preamble='',
                                    input_expr=''):
    if identity is None:
        identity = ''
    module_code = string.Template('''
    ${preamble}
    #define REDUCE(a, b) (${reduce_expr})
    #define POST_MAP(a) (${post_map_expr})

    typedef ${reduce_type} reduce_type;
    extern "C" __global__ void ${name}(${params}) {
      if (_out_clp2_size > 256) {
        CUPY_FOR(_i, _out_size) {
          reduce_type _s = reduce_type(${identity});
          for (int _j = _i, _J = 0;
               _j < _in_size;
               _j += _out_size, _J++) {
            ${input_expr}
            reduce_type _a = ${pre_map_expr};
            _s = REDUCE(_s, _a);
          }
          _raw_out[_i] = POST_MAP(_s);
        }
      } else {
        extern __shared__ reduce_type _sdata_raw[];
        reduce_type *_sdata = _sdata_raw;
        int _tid = threadIdx.x;
        _sdata[_tid] = reduce_type(${identity});
        unsigned int _i = _tid % _out_clp2_size;
        if (_i >= _out_size) return;
        reduce_type _s = reduce_type(${identity});
        int _J_offset = _tid / _out_clp2_size;
        int _j_offset = _J_offset * _out_size;
        int _J_stride = ${block_size} / _out_clp2_size;
        int _j_stride = _J_stride * _out_size;
        for (int _j = _i + _j_offset, _J = _J_offset;
             _j < _in_size;
             _j += _j_stride, _J += _J_stride) {
          ${input_expr}
          reduce_type _a = ${pre_map_expr};
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
        if (_tid >= _out_size) return;
        _raw_out[_i] = POST_MAP(_s);
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
        preamble=preamble,
        input_expr=input_expr)
    module = carray.compile_with_cache(module_code)
    return module.get_function(name)


class simple_reduction_function(object):

    def __init__(self, name, ops, identity=None, preamble='', nout=1):
        self.name = name
        self._ops = ops
        self.identity = identity
        self._preamble = preamble
        self.nout = nout
        self._param_names = [
            '_raw_in', '_raw_out', '_in_size', '_out_size', '_out_clp2_size']

    def __call__(self, a, axis=None, dtype=None, out=None, keepdims=False,
                 allocator=None):
        if not isinstance(a, cupy.ndarray):
            raise TypeError('Input type must be cupy.ndarray')
        internal.check_args_device((a, out))

        if self.identity is None:
            assert a.size != 0

        if axis is None:
            axis = tuple(numpy.arange(len(a.shape)))
        elif isinstance(axis, collections.Iterable):
            axis = tuple(axis)
        else:
            axis = axis,

        if any(ax < -a.ndim or ax >= a.ndim for ax in axis):
            raise ValueError('Axis overrun')
        axis = tuple(ax if ax >= 0 else ax + a.ndim for ax in axis)

        kernel_out_shape = tuple(numpy.delete(a.shape, axis))
        if keepdims:
            out_shape = tuple(1 if i in axis else s
                              for i, s in enumerate(a.shape))
        else:
            out_shape = kernel_out_shape

        in_types, out_types, routine = self._guess_routine([a], dtype)

        if out is not None:
            assert self.nout == 1
            if out.shape != out_shape:
                raise ValueError('Out shape is mismatched')
            out_args = [out]
        else:
            if allocator is None:
                allocator = a.allocator
            out_args = [
                cupy.empty(shape=out_shape, dtype=t, allocator=allocator)
                for t in out_types]

        # TODO(okuta) sort dimension
        trans = axis + tuple(numpy.delete(numpy.arange(len(a.shape)), axis))
        a = a.transpose(trans)

        kernel_out_args = [i.view() for i in out_args]
        for x in kernel_out_args:
            x.shape = kernel_out_shape

        in_size = numpy.int32(a.size)
        out_size = numpy.int32(kernel_out_args[0].size)
        out_clp2_size = numpy.int32(2 ** int.bit_length(int(out_size - 1)))

        inout_args = [a] + kernel_out_args + [in_size, out_size, out_clp2_size]
        for i, x in enumerate(inout_args):
            if isinstance(x, cupy.ndarray):
                inout_args[i] = x.reduced_view()

        params, kernel_args = elementwise._get_kernel_params_args(
            self._param_names, inout_args)
        block_size = 512
        preamble = '''
            typedef {} T; typedef {} in_type;typedef {} in_raw_type;
            {}'''.format(
            elementwise._get_typename(out_types[0]),
            elementwise._get_typename(in_types[0]),
            elementwise._get_typename(a.dtype),
            self._preamble)
        input_expr = 'const in_raw_type in = _raw_in[_j];'
        kern = _make_reduction_function_kernel(
            self.name,
            block_size,
            routine[3],
            params,
            self.identity,
            routine[1],
            routine[0],
            routine[2],
            preamble=preamble,
            input_expr=input_expr)
        shared_mem = 32 * block_size
        if out_clp2_size > 256:
            shared_mem = 0
        # TODO(okuta) set actual size
        kern.linear_launch(max(out_size, block_size), kernel_args,
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
    def __init__(self, in_params, out_type, map_expr, reduce_expr, identity,
                 name='reduce_kernel', options=(), reduce_dims=True,
                 post_map_expr='a', preamble=''):
        self.in_params = elementwise._get_param_info_tuple(in_params, True)
        self.out_params = elementwise._get_param_info_tuple(
            '{} out'.format(out_type))
        self.params = self.in_params + self.out_params + \
            elementwise._get_param_info_tuple(
                'int32 _in_size, int32 _out_size, int32 _out_clp2_size')
        self.identity = identity
        self.reduce_expr = reduce_expr
        self.map_expr = map_expr
        self.name = name
        self.options = options
        self.reduce_dims = reduce_dims
        self.post_map_expr = post_map_expr
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

        allocator = kwargs.get('allocator', None)
        axis = kwargs.get('axis', None)
        keepdims = kwargs.get('keepdims', False)

        if not (len(args) == len(self.in_params) or
                len(args) == len(self.in_params) + len(self.out_params)):
            raise TypeError('Wrong number of arguments for %s' % self.name)
        assert all(i is not None for i in args)
        internal.check_args_device(args)

        out_args = args[len(self.in_params):]
        out = kwargs.pop('out', None)
        if out is not None:
            if len(out_args) == 1:
                raise ValueError('')
            out_args = [out]

        brod = cupy.broadcast(
            *[None if p.raw else a for p, a in zip(self.in_params, args)])
        if all(not isinstance(i, cupy.ndarray) for i in brod.values):
            raise ValueError('Loop size is Undecided')

        if self.identity is None:
            assert brod.size != 0

        if axis is None:
            axis = tuple(numpy.arange(len(brod.shape)))
        elif isinstance(axis, collections.Iterable):
            if any(p.raw for i in self.in_params):
                raise NotImplementedError()
            axis = tuple(axis)
        else:
            if any(p.raw for i in self.in_params):
                raise NotImplementedError()
            axis = axis,

        if any(ax < -brod.nd or ax >= brod.nd for ax in axis):
            raise ValueError('Axis overrun')
        axis = tuple([ax if ax >= 0 else ax + brod.nd for ax in axis].sort())

        in_args = [b if a is None else a for a, b in zip(brod.values, args)]
        kernel_out_shape = tuple(numpy.delete(brod.shape, axis))
        if keepdims:
            out_shape = tuple(1 if i in axis else s
                              for i, s in enumerate(brod.shape))
        else:
            out_shape = kernel_out_shape

        in_types, out_types, types = elementwise._decide_params_type(
            self.in_params, self.out_params,
            elementwise._get_ndarray_dtype(in_args),
            elementwise._get_ndarray_dtype(out_args))

        if len(out_args) == 0:
            if allocator is None:
                allocator = elementwise._get_allocator(in_args)
            out_args = [
                cupy.empty(shape=out_shape, dtype=t, allocator=allocator)
                for t in out_types]
        else:
            if not isinstance(out_args[0], cupy.ndarray):
                raise TypeError(
                    'Output argument type must be cupy.ndarray')
            if out_args[0].shape != out_shape:
                raise ValueError('Output shape error')

        # TODO(okuta) sort dimension
        raw_axis = numpy.arange(len(brod.shape))
        trans = axis + tuple(numpy.delete(raw_axis, axis))

        if not all(i == j for i, j in zip(trans, raw_axis)):
            if any(p.raw for i in self.in_params):
                raise NotImplementedError()
            for i, x in enumerate(in_args):
                if not self.in_params[i].raw and isinstance(x, cupy.ndarray):
                    in_args[i] = x.transpose(trans)

        kernel_out_args = [i.view() for i in out_args]
        for x in kernel_out_args:
            x.shape = kernel_out_shape

        in_size = numpy.int32(brod.size)
        out_size = numpy.int32(kernel_out_args[0].size)
        out_clp2_size = numpy.int32(2 ** int.bit_length(int(out_size - 1)))

        inout_args = in_args + kernel_out_args + [
            numpy.int32(in_size), numpy.int32(out_size),
            numpy.int32(out_clp2_size)]
        param_names = []
        for i, x in enumerate(inout_args):
            name = self.params[i].name
            if isinstance(x, cupy.ndarray):
                inout_args[i] = x.reduced_view()
                if not self.params[i].raw:
                    name = "_raw_" + name
            elif i < len(in_args):
                inout_args[i] = in_types[i].type(x)
            param_names.append(name)

        params, kernel_args = elementwise._get_kernel_params_args(
            param_names, inout_args)
        block_size = 512

        types_preamble = '\n'.join(
            'typedef {} {};'.format(elementwise._get_typename(v), k)
            for k, v in types)
        preamble = types_preamble + '\n' + self.preamble
        input_expr = []
        for a, p in zip(in_args, self.in_params):
            if not p.raw and isinstance(a, cupy.ndarray):
                input_expr.append(
                    'const {t} {n} = _raw_{n}[_j];'.format(
                        t=p.ctype, n=p.name))
        input_expr = '\n'.join(input_expr)

        kern = _make_reduction_function_kernel(
            self.name,
            block_size,
            self.out_params[0].ctype,
            params,
            self.identity,
            self.map_expr,
            self.reduce_expr,
            self.post_map_expr,
            preamble,
            input_expr)
        shared_mem = 32 * block_size
        if out_clp2_size > 256:
            shared_mem = 0
        # TODO(okuta) set actual size
        kern.linear_launch(max(out_size, block_size), kernel_args,
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
        name, _ops, identity=identity, preamble=preamble, nout=len(_ops[0][1]))


_min_max_preamble = '''
struct min_max_st{
    in_raw_type value;
    int index;
    __device__ min_max_st() : index(-1) { }
    __device__ min_max_st(in_raw_type v) : value(v), index(0) { }
    __device__ min_max_st(in_raw_type v, int i) : value(v), index(i) { }
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
    ('my_min(a, b)', 'min_max_st(in)', 'a.value', 'min_max_st'),
    None, _min_max_preamble)

amax = create_reduction_func(
    'cupy_max',
    ['?->?', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'],
    ('my_max(a, b)', 'min_max_st(in)', 'a.value', 'min_max_st'),
    None, _min_max_preamble)

argmin = create_reduction_func(
    'cupy_argmin',
    ['?->l', 'B->l', 'h->l', 'H->l', 'i->l', 'I->l', 'l->l', 'L->l',
     'q->l', 'Q->l', 'e->l', 'f->l', 'd->l'],
    ('my_argmin(a, b)', 'min_max_st(in, _J)', 'a.index', 'min_max_st'),
    None, _min_max_preamble)

argmax = create_reduction_func(
    'cupy_argmax',
    ['?->l', 'B->l', 'h->l', 'H->l', 'i->l', 'I->l', 'l->l', 'L->l',
     'q->l', 'Q->l', 'e->l', 'f->l', 'd->l'],
    ('my_argmax(a, b)', 'min_max_st(in, _J)', 'a.index', 'min_max_st'),
    None, _min_max_preamble)
