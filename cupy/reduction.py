import collections
import string

import numpy

import cupy
from cupy import carray
from cupy import cuda
from cupy import elementwise
from cupy import internal


@cuda.memoize
def _make_reduction_function_kernel(name, block_size,
                                    dtype, temp_type, params,
                                    identity,
                                    reduce_expr,
                                    pre_map_expr='in[i]',
                                    post_map_expr='a',
                                    preamble=''):
    if identity is None:
        identity = ''
    module_code = string.Template('''
    typedef ${dtype} dtype;

    #define REDUCE(a, b) (${reduce_expr})
    #define POST_MAP(a) (${post_map_expr})
    ${preamble}
    typedef ${temp_type} temp_type;
    extern "C" __global__ void ${name}(${params}) {
      if (_out_clp2_size > 256) {
        CUPY_FOR(_i, out_size) {
          temp_type _s = temp_type(${identity});
          for (int i = _i, I = 0;
               i < in_size;
               i += out_size, I++) {
              temp_type _a = ${pre_map_expr};
              _s = REDUCE(_s, _a);
          }
          out[_i] = POST_MAP(_s);
        }
      } else {
        extern __shared__ temp_type _sdata_raw[];
        temp_type *_sdata = _sdata_raw;//[${block_size}];
        int _tid = threadIdx.x;
        _sdata[_tid] = temp_type(${identity});
        unsigned int _i = _tid % _out_clp2_size;
        if (_i >= out_size) return;
        temp_type _s = temp_type(${identity});
        int _I_offset = _tid / _out_clp2_size;
        int _i_offset = _I_offset * out_size;
        int _I_stride = ${block_size} / _out_clp2_size;
        int _i_stride = _I_stride * out_size;
        for (int i = _i + _i_offset, I = _I_offset;
             i < in_size;
             i += _i_stride, I += _I_stride) {
          temp_type _a = ${pre_map_expr};
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
        if (_tid >= out_size) return;
        out[_i] = POST_MAP(_s);
      }
    }''').substitute(
        name=name,
        block_size=block_size,
        dtype=dtype,
        temp_type=temp_type,
        params=params,
        identity=identity,
        reduce_expr=reduce_expr,
        pre_map_expr=pre_map_expr,
        post_map_expr=post_map_expr,
        preamble=preamble)
    module = carray.compile_with_cache(module_code)
    return module.get_function(name)


class simple_reduction_function(object):

    def __init__(self, name, ops, identity=None, preamble='',
                 temp_type='dtype', nout=1):
        self.name = name
        self._ops = ops
        self.identity = identity
        self._preamble = preamble
        self._temp_type = temp_type
        self.nout = nout
        self._param_names = [
            'in', 'out', 'in_size', 'out_size', '_out_clp2_size']

    def __call__(self, a, axis=None, dtype=None, out=None, keepdims=False,
                 allocator=None):
        if not isinstance(a, cupy.ndarray):
            raise TypeError('')
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
                raise ValueError('')
            out_args = [out]
        else:
            if allocator is None:
                allocator = a.allocator
            out_args = [
                cupy.empty(shape=out_shape, dtype=t, allocator=allocator)
                for t in out_types]

        # TODO(okuta) sort dimension
        a = a.transpose(
            axis + tuple(numpy.delete(numpy.arange(len(a.shape)), axis)))
        kernel_out_args = [i.view() for i in out_args]
        for x in kernel_out_args:
            x.shape = kernel_out_shape

        in_size = a.size
        out_size = kernel_out_args[0].size
        out_clp2_size = numpy.uint32(2 ** int.bit_length(int(out_size - 1)))

        inout_args = [a] + kernel_out_args + [in_size, out_size, out_clp2_size]
        for i, x in enumerate(inout_args):
            if isinstance(x, cupy.ndarray):
                inout_args[i] = x.reduced_view()

        params, kernel_args = elementwise._get_kernel_params_args(
            self._param_names, inout_args)
        block_size = 512
        kern = _make_reduction_function_kernel(
            self.name,
            block_size,
            elementwise._get_typename(dtype),
            self._temp_type,
            params,
            self.identity,
            routine[0],
            routine[1],
            routine[2],
            preamble=self._preamble)
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
                if all(numpy.dtype(t) == dtype for t in out_types):
                    return in_types, out_types, routine
        raise TypeError('Wrong type of arguments for %s' % self.name)


class ReductionKernel(object):

    def __init__(self, out_dtype, param_names, identity, reduce_expr,
                 map_expr, post_map_expr='a', name='reduce_kernel',
                 options=[], preamble=''):
        self.out_dtype = out_dtype
        self.param_names = ('out',) + tuple(param_names) + (
            'in_size', 'out_size', '_out_clp2_size')
        self.identity = identity
        self.reduce_expr = reduce_expr
        self.map_expr = map_expr
        self.name = name
        self.options = []
        self.preamble = preamble

    def __call__(self, *args, **kwargs):
        reduce_dims = kwargs.pop('reduce_dims', True)
        in_size = None
        for i in args:
            if isinstance(i, cupy.ndarray):
                in_size = i.size
                break
        assert in_size is not None
        args = list(args)
        for i, x in enumerate(args):
            if isinstance(x, cupy.ndarray):
                if reduce_dims:
                    args[i] = x.reduced_view()

        out = cupy.empty((), dtype=self.out_dtype)
        args = [out] + args + [
            numpy.int32(in_size), numpy.int32(1), numpy.int32(1)]
        assert len(self.param_names) == len(args)
        internal.check_args_device(args)
        params, kernel_args = elementwise._get_kernel_params_args(
            self.param_names, args)
        block_size = 512
        dtype = elementwise._get_typename(args[0].dtype)
        kernel = _make_reduction_function_kernel(
            self.name, block_size, dtype, dtype, params, self.identity,
            self.reduce_expr, self.map_expr, 'a', self.preamble)
        shared_mem = 32 * block_size
        # TODO(okuta) set actual size
        kernel.linear_launch(block_size, kernel_args,
                             shared_mem=shared_mem,
                             block_max_size=block_size)

        return out


def create_reduction_func(name, ops, routine=None, identity=None,
                          preamble='', temp_type='dtype'):
    _ops = []
    for t in ops:
        if not isinstance(t, tuple):
            typ = t
            rt = routine
        else:
            typ, rt = t
            rt = (i or j for i, j in zip(routine, rt))

        types = typ.split('->')
        if len(types) == 1:
            in_types = out_types = tuple(types)
        else:
            in_types, out_types = map(tuple, types)
        in_types = [numpy.dtype(t).type for t in in_types]
        out_types = [numpy.dtype(t).type for t in out_types]
        _ops.append((in_types, out_types, rt))

    return simple_reduction_function(
        name, _ops, identity=identity, preamble=preamble,
        temp_type=temp_type,  nout=len(_ops[0][1]))


_min_max_preamble = '''
struct my_struct{
    dtype value;
    int index;
    __device__ my_struct() : index(-1) { }
    __device__ my_struct(dtype v) : value(v), index(0) { }
    __device__ my_struct(dtype v, int i) : value(v), index(i) { }
};
__device__ my_struct my_min(my_struct& a, my_struct& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return my_struct(min(a.value, b.value));
}
__device__ my_struct my_max(my_struct& a, my_struct& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return my_struct(max(a.value, b.value));
}
__device__ my_struct my_argmin(my_struct& a, my_struct& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return (a.value <= b.value) ? a : b;
}
__device__ my_struct my_argmax(my_struct& a, my_struct& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return (a.value >= b.value) ? a : b;
}'''


amin = create_reduction_func(
    'cupy_min',
    ['?->?', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'],
    ('my_min(a, b)', 'my_struct((dtype)in[i])', 'a.value'),
    None, _min_max_preamble, temp_type='my_struct')

amax = create_reduction_func(
    'cupy_max',
    ['?->?', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'],
    ('my_max(a, b)', 'my_struct((dtype)in[i])', 'a.value'),
    None, _min_max_preamble, temp_type='my_struct')

argmin = create_reduction_func(
    'cupy_argmin',
    ['?->l', 'B->l', 'h->l', 'H->l', 'i->l', 'I->l', 'l->l', 'L->l',
     'q->l', 'Q->l', 'e->l', 'f->l', 'd->l'],
    ('my_argmin(a, b)', 'my_struct((dtype)in[i], I)', 'a.index'),
    None, _min_max_preamble, temp_type='my_struct')

argmax = create_reduction_func(
    'cupy_argmax',
    ['?->l', 'B->l', 'h->l', 'H->l', 'i->l', 'I->l', 'l->l', 'L->l',
     'q->l', 'Q->l', 'e->l', 'f->l', 'd->l'],
    ('my_argmax(a, b)', 'my_struct((dtype)in[i], I)', 'a.index'),
    None, _min_max_preamble, temp_type='my_struct')
