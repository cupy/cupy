import string

from cupy._core cimport _kernel
from cupy._core.core cimport _ndarray_base, _internal_ascontiguousarray
from cupy._core._cuda_compute_common cimport (_get_cuda_compute,
                                              cached_algorithm)
from cupy._core._cuda_compute_common import _make_raw_ops

import numpy

import cupy
from cupy._core._dtype import make_aligned_dtype
from cupy._core._scalar import format_type_decls
from cupy._core._scalar import get_typename


cdef str _get_reduce_op_src(str reduce_expr):
    return string.Template('''
extern "C" __device__ void op(void* _a, void* _b, void* _ret) {
    _type_reduce a = *static_cast<const _type_reduce*>(_a);
    _type_reduce b = *static_cast<const _type_reduce*>(_b);
    *static_cast<_type_reduce*>(_ret) = (${reduce_expr});
}
''').substitute(reduce_expr=reduce_expr)


cdef str _get_input_map_src(str map_expr):
    return string.Template('''
extern "C" __device__ void map(void* _in, void* _ret) {
    const type_in0_raw in0 = *static_cast<const type_in0_raw*>(_in);
    *static_cast<_type_reduce*>(_ret) =
        static_cast<_type_reduce>(${map_expr});
}
''').substitute(map_expr=map_expr)


cdef str _get_zip_map_src(str map_expr):
    # member order matches ZipIterator(ids, d_in) in _try_reduction
    return string.Template('''
struct _zip_in { IndexT _idx; type_in0_raw _val; };
extern "C" __device__ void map(void* _in, void* _ret) {
    const _zip_in _z = *static_cast<const _zip_in*>(_in);
    const type_in0_raw in0 = _z._val;
    const IndexT _J = _z._idx;
    *static_cast<_type_reduce*>(_ret) =
        static_cast<_type_reduce>(${map_expr});
}
''').substitute(map_expr=map_expr)


cdef str _get_output_map_src(str post_map_expr):
    return string.Template('''
extern "C" __device__ void post(void* _acc, void* _ret) {
    _type_reduce a = *static_cast<const _type_reduce*>(_acc);
    type_out0_raw _out;
    type_out0_raw &out0 = _out;
    (${post_map_expr});
    *static_cast<type_out0_raw*>(_ret) = _out;
}
''').substitute(post_map_expr=post_map_expr)


cdef str _get_offset_ops_src():
    # segment offsets for the segmented reduce: start = i * seg_size and
    # end = start + seg_size, computed from a counting/constant iterator zip
    return """
extern "C" __device__ void mul_offset(void* a, void* result) {
    const long long* f = static_cast<const long long*>(a);
    *static_cast<long long*>(result) = f[0] * f[1];
}
extern "C" __device__ void add_offset(void* a, void* result) {
    const long long* f = static_cast<const long long*>(a);
    *static_cast<long long*>(result) = f[0] + f[1];
}
"""


cdef str _get_kernel_prelude(_kernel._TypeMap type_map, str preamble,
                             str reduce_type, acc_dtype):
    type_decls = set()
    typedefs = type_map.get_typedef_code(type_decls)
    tpl = string.Template('''${type_decls}${typedefs}
${preamble}
typedef ${reduce_type} _type_reduce;
static_assert(sizeof(_type_reduce) == ${acc_size},
              "accumulator layout must match the h_init dtype");''')
    return tpl.substitute(
        type_decls=format_type_decls(type_decls), typedefs=typedefs,
        preamble=preamble, reduce_type=reduce_type,
        acc_size=acc_dtype.itemsize)


cpdef _can_use_cuda_compute_reduction(
        list in_args, list out_args, tuple reduce_axis, tuple out_axis):
    cdef _ndarray_base input_array

    if _get_cuda_compute() is None:
        return False

    # support reductions with only 1 input and 1 output
    if len(in_args) != 1 or len(out_args) != 1:
        return False

    input_array = in_args[0]

    if not out_args[0]._c_contiguous:
        return False

    if len(out_axis) != 0:
        if not input_array._c_contiguous:
            return False
        if (tuple(out_axis) + tuple(reduce_axis)
                != tuple(range(input_array.ndim))):
            return False
        return True

    # TODO: add support for StridedIterator
    if not (input_array._c_contiguous or input_array._f_contiguous):
        return False

    return True


_CTYPE_TO_DTYPE = {
    get_typename(numpy.dtype(ch)): numpy.dtype(ch)
    for ch in '?bBhHiIlLqQefdFD'}


_IDENTITY_VALUES = {'0': 0, '1': 1, 'true': True, 'false': False}


cdef _try_accumulator(str reduce_type, str identity, in_dtype,
                      out_dtype, index_dtype):
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

    # the structured accumulators mirror device structs, so their member
    # offsets must follow the device alignment (thrust::complex<double> is
    # __align__(16) while numpy aligns complex128 to 8)
    if reduce_type == 'min_max_st<type_in0_raw>':
        # complex64 accumulators fail to build in cuda.compute:
        # "Host generated and JIT compiled reduce policy mismatch"
        if in_dtype.kind == 'c':
            return None
        acc = make_aligned_dtype(
            [('value', in_dtype), ('index', index_dtype)])
        h_init = numpy.zeros((), dtype=acc)
        h_init['index'] = -1
        return acc, h_init

    if reduce_type == 'nanmean_st<type_out0_raw>':
        if out_dtype.kind == 'c':
            return None
        acc = make_aligned_dtype(
            [('value', out_dtype), ('count', numpy.int64)])
        return acc, numpy.zeros((), dtype=acc)

    return None


def _try_reduction(_ndarray_base input_array, _ndarray_base out,
                   str map_expr, str reduce_expr, str post_map_expr,
                   str reduce_type, _kernel._TypeMap type_map,
                   str identity, str preamble, compute_opkind):
    """Turn the routine's C++ expressions into cuda.compute ops
    and iterators.

    Returns (d_in, d_out, build_in, build_out, op, h_init, ops, build_key)
    or None
    """
    compute = _get_cuda_compute()

    # a segmented reduction needs _J to be the index within each
    # segment (_J % seg_size). seg_size is a host-side value, which
    # ops passed to transform iterators do not receive yet
    # (fixed in NVIDIA/cccl#11213)
    # TODO: pass seg_size as RawOp state once a cuda-cccl release has the fix
    if '_J' in map_expr and out.size > 1:
        return None

    # post_map_expr needs a host-side value (e.g. for mean: the
    # divisor _in_ind.size() / _out_ind.size()), same limitation as above
    # TODO: pass the input and output sizes as RawOp state once a cuda-cccl
    # release has the fix
    if '_in_ind' in post_map_expr or '_out_ind' in post_map_expr:
        return None

    # complex -> real load cast fails to compile in CuPy
    if (map_expr == 'in0' and input_array.dtype.kind == 'c'
            and out.dtype.kind != 'c'):
        return None

    index_dtype = numpy.dtype(dict(type_map._pairs).get('IndexT', 'q'))
    acc = _try_accumulator(reduce_type, identity, input_array.dtype,
                           out.dtype, index_dtype)
    if acc is None:
        return None
    acc_dtype, h_init = acc

    prelude = _get_kernel_prelude(type_map, preamble, reduce_type,
                                  acc_dtype)

    if '_J' in map_expr and not input_array._c_contiguous:
        # in NumPy the indices are always generated based on a C-order
        # array, so _J must count in C order
        d_in = _internal_ascontiguousarray(input_array).ravel()
    else:
        d_in = input_array.ravel(order='A')
    out_flat = out.ravel()

    # every op linked into the reduction kernel comes from one module, so
    # collect the sources first and compile them together
    to_complex_acc = input_array.dtype != acc_dtype and acc_dtype.kind == 'c'
    struct_acc = acc_dtype.kind == 'V'
    use_zip_map = '_J' in map_expr
    use_map = use_zip_map or map_expr != 'in0' or to_complex_acc or struct_acc
    use_opkind = compute_opkind is not None and acc_dtype.kind in 'biuf'
    use_post = not (post_map_expr in ('out0 = a', 'out0 = type_out0_raw(a)')
                    and acc_dtype == out.dtype)

    src = prelude
    names = []
    if use_map:
        src += (_get_zip_map_src(map_expr) if use_zip_map
                else _get_input_map_src(map_expr))
        names.append('map')
    if not use_opkind:
        src += _get_reduce_op_src(reduce_expr)
        names.append('op')
    if use_post:
        src += _get_output_map_src(post_map_expr)
        names.append('post')
    if out.size > 1:
        src += _get_offset_ops_src()
        names += ['mul_offset', 'add_offset']
    ops = _make_raw_ops(src, tuple(names)) if names else {}

    build_key = (out.size > 1, src, tuple(names), use_zip_map, use_map,
                 use_post, compute_opkind if use_opkind else None,
                 input_array.dtype.str, out.dtype.str, acc_dtype.str)

    acc_type_descriptor = compute.types.from_numpy_dtype(acc_dtype)

    build_in = compute.ProxyArray(input_array.dtype)
    if use_zip_map:
        ids = compute.CountingIterator(numpy.zeros((), dtype=index_dtype))
        d_in = compute.TransformIterator(
            compute.ZipIterator(ids, d_in), ops['map'], acc_type_descriptor)
        build_in = d_in
    elif use_map:
        d_in = compute.TransformIterator(d_in, ops['map'],
                                         acc_type_descriptor)
        build_in = d_in

    if use_opkind:
        op = getattr(compute.OpKind, compute_opkind)
    else:
        op = ops['op']

    build_out = compute.ProxyArray(out.dtype)
    if use_post:
        d_out = compute.TransformOutputIterator(
            out_flat, ops['post'], acc_type_descriptor)
        build_out = d_out
    else:
        d_out = out_flat

    return d_in, d_out, build_in, build_out, op, h_init, ops, build_key


def _cuda_compute_reduce(_ndarray_base input_array, _ndarray_base out,
                         str map_expr, str reduce_expr, str post_map_expr,
                         str reduce_type, type_map, str identity,
                         str preamble, compute_opkind, stream):
    compute = _get_cuda_compute()
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    build_cuda_compute_reduce = _try_reduction(
        input_array, out, map_expr, reduce_expr, post_map_expr, reduce_type,
        type_map, identity, preamble, compute_opkind)
    if build_cuda_compute_reduce is None:
        return False
    (d_in, d_out, build_in, build_out, op, h_init, ops,
     build_key) = build_cuda_compute_reduce

    if out.size > 1:
        num_segments = out.size
        # NOTE: input_array.size == out.size * (elements per output)
        seg_size = input_array.size // num_segments
        # TODO: use CUB's fixed-size segmented reduce when exposed
        # in cuda.compute
        ids = compute.CountingIterator(numpy.int64(0))
        size = compute.ConstantIterator(numpy.int64(seg_size))
        start = compute.TransformIterator(
            compute.ZipIterator(ids, size), ops['mul_offset'],
            value_type=compute.types.int64)
        end = compute.TransformIterator(
            compute.ZipIterator(start, size), ops['add_offset'],
            value_type=compute.types.int64)
        reducer = cached_algorithm(
            'segmented_reduce', build_key, repr(build_key),
            lambda: compute.make_segmented_reduce(
                d_in=build_in, d_out=build_out, op=op, h_init=h_init,
                start_offsets_in=start, end_offsets_in=end))
        tmp_size = reducer(temp_storage=None, d_in=d_in, d_out=d_out,
                           num_segments=num_segments, op=op,
                           h_init=h_init, start_offsets_in=start,
                           end_offsets_in=end, max_segment_size=seg_size)
        d_tmp = cupy.empty(tmp_size, dtype=numpy.uint8)
        reducer(temp_storage=d_tmp, d_in=d_in, d_out=d_out,
                num_segments=num_segments, op=op, h_init=h_init,
                start_offsets_in=start, end_offsets_in=end,
                max_segment_size=seg_size, stream=stream)
        return True

    reducer = cached_algorithm(
        'reduce', build_key, repr(build_key),
        lambda: compute.make_reduce_into(
            d_in=build_in, d_out=build_out, op=op, h_init=h_init))
    tmp_size = reducer(temp_storage=None, d_in=d_in, d_out=d_out,
                       num_items=input_array.size, op=op, h_init=h_init)
    d_tmp = cupy.empty(tmp_size, dtype=numpy.uint8)
    reducer(temp_storage=d_tmp, d_in=d_in, d_out=d_out,
            num_items=input_array.size, op=op, h_init=h_init, stream=stream)
    return True


cdef bint _try_to_call_cuda_compute_reduction(
        self, list in_args, list out_args, stream, str map_expr,
        str reduce_expr, str post_map_expr, str reduce_type, type_map,
        tuple reduce_axis, tuple out_axis, _ndarray_base ret) except *:
    """Try to use cuda.compute (CUB DeviceReduce).

    Updates `ret` and returns a boolean value
    """
    if not _can_use_cuda_compute_reduction(
            in_args, out_args, reduce_axis, out_axis):
        return False

    return _cuda_compute_reduce(
        in_args[0], ret, map_expr, reduce_expr, post_map_expr,
        reduce_type, type_map, self.identity, self.preamble,
        getattr(self, 'compute_opkind', None), stream)
