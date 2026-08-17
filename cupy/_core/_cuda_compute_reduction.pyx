import string

from cupy._core cimport _kernel
from cupy._core._carray cimport shape_t
from cupy._core.core cimport _ndarray_base

import numpy

import cupy
from cupy import _util
from cupy._core import core
from cupy._core._scalar import format_type_decls
from cupy._core._scalar import get_typename
from cupy.cuda import compiler


# lazy import: avoid importing cuda.compute whenever cupy is imported
@_util.memoize()
def _get_cuda_compute():
    try:
        from cuda import compute
    except ImportError:
        return None
    return compute


def _compile_cpp_to_ltoir(str src):
    options = core.assemble_cupy_compiler_options(())
    return compiler._compile_module_with_cache(src, options, to_ltoir=True)


@_util.memoize(for_each_device=True)
def _make_raw_op(str src, str name):
    ltoir = _compile_cpp_to_ltoir(src)
    return _get_cuda_compute().op.RawOp(ltoir=ltoir, name=name)


_REDUCE_OP_SRC = string.Template('''
${prelude}
extern "C" __device__ void op(void* _a, void* _b, void* _ret) {
    _type_reduce a = *static_cast<const _type_reduce*>(_a);
    _type_reduce b = *static_cast<const _type_reduce*>(_b);
    *static_cast<_type_reduce*>(_ret) = (${reduce_expr});
}
''')


_MAP_OP_SRC = string.Template('''
${prelude}
extern "C" __device__ void map(void* _in, void* _ret) {
    const type_in0_raw in0 = *static_cast<const type_in0_raw*>(_in);
    *static_cast<_type_reduce*>(_ret) =
        static_cast<_type_reduce>(${map_expr});
}
''')


_POST_OP_SRC = string.Template('''
${prelude}
extern "C" __device__ void post(void* _acc, void* _ret) {
    _type_reduce a = *static_cast<const _type_reduce*>(_acc);
    type_out0_raw _out;
    type_out0_raw &out0 = _out;
    (${post_map_expr});
    *static_cast<type_out0_raw*>(_ret) = _out;
}
''')


_MUL_OFFSET_SRC = """
extern "C" __device__ void mul_offset(void* a, void* result) {
    const long long* f = static_cast<const long long*>(a);
    *static_cast<long long*>(result) = f[0] * f[1];
}
"""


_ADD_OFFSET_SRC = """
extern "C" __device__ void add_offset(void* a, void* result) {
    const long long* f = static_cast<const long long*>(a);
    *static_cast<long long*>(result) = f[0] + f[1];
}
"""


_PRELUDE = string.Template('''${type_decls}${typedefs}
${preamble}
typedef ${reduce_type} _type_reduce;
static_assert(sizeof(_type_reduce) == ${acc_size},
              "accumulator layout must match the h_init dtype");''')


_CTYPE_TO_DTYPE = {
    get_typename(numpy.dtype(ch)): numpy.dtype(ch)
    for ch in '?bBhHiIlLqQefdFD'}


_IDENTITY_VALUES = {'0': 0, '1': 1, 'true': True, 'false': False}


cpdef _can_use_cuda_compute_reduction(
        list in_args, list out_args, tuple reduce_axis, tuple out_axis):
    cdef _ndarray_base x

    if _get_cuda_compute() is None:
        return False

    # support reductions with only 1 input and 1 output
    if len(in_args) != 1 or len(out_args) != 1:
        return False

    x = in_args[0]

    if len(out_axis) != 0:
        if not x._c_contiguous:
            return False
        if tuple(out_axis) + tuple(reduce_axis) != tuple(range(x.ndim)):
            return False
        return True

    # TODO: add support for StridedIterator
    if not (x._c_contiguous or x._f_contiguous):
        return False

    return True


def _try_accumulator(str reduce_type, str identity, in_dtype, out_dtype):
    """Map the kernel's _type_reduce C type onto a numpy dtype and build
    the matching h_init value for cuda.compute.

    Returns (acc_dtype, h_init) or None
    """
    dt = _CTYPE_TO_DTYPE.get(reduce_type)
    if dt is not None:
        value = _IDENTITY_VALUES.get(identity)
        if value is None:
            return None
        return dt, numpy.full((), value, dtype=dt)

    return None


def _try_reduction(x, out, str map_expr, str reduce_expr,
                   str post_map_expr, str reduce_type,
                   _kernel._TypeMap type_map, str identity,
                   str preamble, compute_opkind):
    """Turn the routine's C++ expressions into cuda.compute ops
    and iterators.

    Returns (d_in, d_out, op, h_init) or None
    """
    compute = _get_cuda_compute()

    # fp16 input needs a fp32 accumulator with a converting store
    # to fp16 output
    # relevant: get_accumulator_type TODO in cuda.compute reduce.cu
    if numpy.dtype('float16') in (x.dtype, out.dtype):
        return None

    # TODO: support struct accumulators (e.g. (value, index) pair)
    if '_J' in map_expr:
        return None

    # post_map_expr needs a host-side value (e.g. for mean: the
    # divisor _in_ind.size() / _out_ind.size()). Raw ops can carry
    # state bytes but ops passed to transform iterators never receive them
    if '_in_ind' in post_map_expr or '_out_ind' in post_map_expr:
        return None

    if map_expr == 'in0' and x.dtype.kind == 'c' and out.dtype.kind != 'c':
        return None

    acc = _try_accumulator(
        reduce_type, identity, x.dtype, out.dtype)
    if acc is None:
        return None
    acc_dtype, h_init = acc
    acc_type = compute.types.from_numpy_dtype(acc_dtype)

    type_decls = set()
    typedefs = type_map.get_typedef_code(type_decls)
    prelude = _PRELUDE.substitute(
        type_decls=format_type_decls(type_decls), typedefs=typedefs,
        preamble=preamble, reduce_type=reduce_type,
        acc_size=acc_dtype.itemsize)

    d_in = x.ravel(order='A')
    build_in = compute.ProxyArray(x.dtype)
    to_complex_acc = x.dtype != acc_dtype and acc_dtype.kind == 'c'
    if map_expr != 'in0' or to_complex_acc:
        map_op = _make_raw_op(_MAP_OP_SRC.substitute(
            prelude=prelude, map_expr=map_expr), 'map')
        d_in = compute.TransformIterator(d_in, map_op, acc_type)
        build_in = d_in

    if compute_opkind is not None and acc_dtype.kind in 'biuf':
        op = getattr(compute.OpKind, compute_opkind)
    else:
        op = _make_raw_op(_REDUCE_OP_SRC.substitute(
            prelude=prelude, reduce_expr=reduce_expr), 'op')

    out_flat = out.ravel()
    build_out = compute.ProxyArray(out.dtype)
    if (post_map_expr in ('out0 = a', 'out0 = type_out0_raw(a)')
            and acc_dtype == out.dtype):
        d_out = out_flat
    else:
        post_op = _make_raw_op(_POST_OP_SRC.substitute(
            prelude=prelude, post_map_expr=post_map_expr), 'post')
        d_out = compute.TransformOutputIterator(out_flat, post_op,
                                                acc_type)
        build_out = d_out

    return d_in, d_out, build_in, build_out, op, h_init


def _cuda_compute_reduce(x, out, str map_expr, str reduce_expr,
                         str post_map_expr, str reduce_type, type_map,
                         str identity, str preamble, compute_opkind):
    compute = _get_cuda_compute()
    build_cuda_compute_reduce = _try_reduction(
        x, out, map_expr, reduce_expr, post_map_expr, reduce_type,
        type_map, identity, preamble, compute_opkind)
    if build_cuda_compute_reduce is None:
        return False
    d_in, d_out, build_in, build_out, op, h_init = build_cuda_compute_reduce

    if out.size > 1:
        num_segments = out.size
        # NOTE: x.size == out.size * (elements reduced per output)
        seg_size = x.size // num_segments
        ids = compute.CountingIterator(numpy.int64(0))
        size = compute.ConstantIterator(numpy.int64(seg_size))
        start = compute.TransformIterator(
            compute.ZipIterator(ids, size),
            _make_raw_op(_MUL_OFFSET_SRC, 'mul_offset'),
            value_type=compute.types.int64)
        end = compute.TransformIterator(
            compute.ZipIterator(start, size),
            _make_raw_op(_ADD_OFFSET_SRC, 'add_offset'),
            value_type=compute.types.int64)
        reducer = compute.make_segmented_reduce(
            d_in=build_in, d_out=build_out, op=op, h_init=h_init,
            start_offsets_in=start, end_offsets_in=end)
        tmp_size = reducer(temp_storage=None, d_in=d_in, d_out=d_out,
                           num_segments=num_segments, op=op,
                           h_init=h_init, start_offsets_in=start,
                           end_offsets_in=end)
        d_tmp = cupy.empty(tmp_size, dtype=numpy.uint8)
        reducer(temp_storage=d_tmp, d_in=d_in, d_out=d_out,
                num_segments=num_segments, op=op, h_init=h_init,
                start_offsets_in=start, end_offsets_in=end,
                max_segment_size=seg_size,
                stream=cupy.cuda.get_current_stream())
        return True

    reducer = compute.make_reduce_into(
        d_in=build_in, d_out=build_out, op=op, h_init=h_init)
    tmp_size = reducer(temp_storage=None, d_in=d_in, d_out=d_out,
                       num_items=x.size, op=op, h_init=h_init)
    d_tmp = cupy.empty(tmp_size, dtype=numpy.uint8)
    reducer(temp_storage=d_tmp, d_in=d_in, d_out=d_out,
            num_items=x.size, op=op, h_init=h_init,
            stream=cupy.cuda.get_current_stream())
    return True


cdef bint _try_to_call_cuda_compute_reduction(
        self, list in_args, list out_args, const shape_t& a_shape,
        stream, str map_expr, str reduce_expr, str post_map_expr,
        str reduce_type, type_map, tuple reduce_axis, tuple out_axis,
        const shape_t& out_shape, _ndarray_base ret) except *:
    """Try to use cuda.compute (CUB DeviceReduce).

    Updates `ret` and returns a boolean value
    """
    if not _can_use_cuda_compute_reduction(
            in_args, out_args, reduce_axis, out_axis):
        return False

    return _cuda_compute_reduce(
        in_args[0], ret, map_expr, reduce_expr, post_map_expr,
        reduce_type, type_map, self.identity, self.preamble,
        getattr(self, 'compute_opkind', None))
