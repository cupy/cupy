"""Last-axis merge sort using cuda.compute, with CuPy-managed caches.

This module provides an optional sorting accelerator. It relies on the current
CUB merge-sort implementation's stability: SortKeysCopy and StableSortKeysCopy
share the same implementation. cuda.compute does not expose a separate stable
API, so tests guard the preservation of signed zeros and NaN payloads.
"""
from __future__ import annotations

from textwrap import dedent

import numpy

import cupy
from cupy import _util
from cupy._core import core
from cupy._core._cuda_compute_common import _get_cuda_compute
from cupy._core._cuda_compute_common import cached_algorithm
from cupy._core._scalar import format_type_decls, get_typename
from cupy.cuda import memory


_SUPPORTED_DTYPES = frozenset(
    numpy.dtype(ch).name for ch in '?bBhHiIlLqQefdFD')
_COMPLEX_TYPES = {
    "complex64": "Complex<float>", "complex128": "Complex<double>",
}

_INDEX_DTYPE = numpy.dtype(numpy.intp)
_LEXSORT_STATE_DTYPE = numpy.dtype([
    ('keys', numpy.uintp),
    ('num_keys', numpy.intp),
    ('row_width', numpy.intp),
], align=True)


class _KeyBuffer:
    """Structured device storage, since CuPy does not support struct dtypes."""

    def __init__(self, size, dtype):
        self.dtype = dtype
        self.size = size
        self.shape = (size,)
        self.ndim = 1
        alignment = dtype.alignment
        storage_size = size * dtype.itemsize + alignment - 1
        self._storage = cupy.empty(storage_size, dtype=cupy.uint8)
        _data = self._storage.data
        offset = (-_data.ptr) % alignment
        self.data = _data + offset
        self.__cuda_array_interface__ = {
            "shape": self.shape, "strides": None, "typestr": dtype.str,
            "descr": dtype.descr, "data": (self.data.ptr, False),
            "version": 3,
        }


def _key_dtype(dtype):
    return numpy.dtype([('row', numpy.intp), ('value', dtype)], align=True)


def _comparison_source(dtype_name, descending):
    # The float16 comparison overload below needs CuPy's wrapper definition
    # even when T is another type. Share scalar names with the other kernels.
    type_decls = {
        '#include <cupy/float16.cuh>',
        '#include <cuda/std/cstdint>',
    }
    cpp_type = _COMPLEX_TYPES.get(dtype_name)
    if cpp_type is None:
        cpp_type = get_typename(numpy.dtype(dtype_name), type_decls)
    direction = 'true' if descending else 'false'
    source_prolog = (format_type_decls(type_decls) +
                     f'constexpr bool descending = {direction};\n')
    source_body = r'''
    // CuPy complex storage is two adjacent real components. Keep the native
    // component alignment (rather than CUDA float2/double2 vector alignment).
    template<class R> struct Complex { R real, imag; };

    template<class T>
    __device__ bool numeric_strict_cmp(T a, T b) {
        return descending ? b < a : a < b;
    }
    template<class T>
    __device__ bool value_strict_cmp(T a, T b) {
        return numeric_strict_cmp(a, b);
    }

    template<class R>
    __device__ bool real_strict_cmp(R a, R b) {
        const bool an = a != a, bn = b != b;
        return an ? false : (bn || numeric_strict_cmp(a, b));
    }
    __device__ bool value_strict_cmp(float a, float b) {
        return real_strict_cmp(a, b);
    }
    __device__ bool value_strict_cmp(double a, double b) {
        return real_strict_cmp(a, b);
    }
    __device__ bool value_strict_cmp(float16 a, float16 b) {
        return real_strict_cmp(float(a), float(b));
    }

    template<class R>
    __device__ bool value_strict_cmp(Complex<R> a, Complex<R> b) {
        // NumPy categories, ordered before comparing the non-NaN components:
        // (R,R), (R,NaN), (NaN,R), (NaN,NaN).
        const bool arn = a.real != a.real, ain = a.imag != a.imag;
        const bool brn = b.real != b.real, bin = b.imag != b.imag;
        const int ac = 2 * int(arn) + int(ain);
        const int bc = 2 * int(brn) + int(bin);
        if (ac != bc) return ac < bc;
        if (!arn && a.real != b.real)
            return numeric_strict_cmp(a.real, b.real);
        return !ain && numeric_strict_cmp(a.imag, b.imag);
    }
    '''
    source_epilog = (
        f'using T = {cpp_type};\n'
        f'static_assert(sizeof(intptr_t) == {_INDEX_DTYPE.itemsize});\n'
        f'static_assert(alignof(intptr_t) == {_INDEX_DTYPE.alignment});\n')
    return source_prolog + dedent(source_body) + source_epilog


def _layout_assertions(cpp_type, dtype):
    """Check that a C++ struct matches its NumPy storage layout."""
    source = (
        f'static_assert(sizeof({cpp_type}) == {dtype.itemsize});\n'
        f'static_assert(alignof({cpp_type}) == {dtype.alignment});\n')
    for name in dtype.names:
        source += (
            f'static_assert(__builtin_offsetof({cpp_type}, {name}) '
            f'== {dtype.fields[name][1]});\n')
    return source


@_util.memoize(for_each_device=True)
def _get_specialization(dtype, descending, multi_row):
    """Compile ops once per device; array sizes are runtime state only."""
    layout = _key_dtype(dtype) if multi_row else dtype
    source = _comparison_source(dtype.name, descending)
    if multi_row:
        source += 'struct Key { intptr_t row; T value; };\n'
        source += _layout_assertions('Key', layout)
        source += dedent(r'''
            __device__ bool key_strict_cmp(const Key& a, const Key& b) {
                return a.row != b.row ? a.row < b.row :
                    value_strict_cmp(a.value, b.value);
            }
            extern "C" __device__ void row_id(
                    void* state, const void* input, void* result) {
                *static_cast<intptr_t*>(result) =
                    *static_cast<const intptr_t*>(input) /
                    *static_cast<const intptr_t*>(state);
            }
            extern "C" __device__ void column_id(
                    void* state, const void* input, void* result) {
                *static_cast<intptr_t*>(result) =
                    *static_cast<const intptr_t*>(input) %
                    *static_cast<const intptr_t*>(state);
            }
        ''')
    else:
        source += dedent(r'''
            using Key = T;
            __device__ bool key_strict_cmp(const Key& a, const Key& b) {
                return value_strict_cmp(a, b);
            }
        ''')
    source += dedent(r'''
        extern "C" __device__ void strict_cmp(
                const void* lhs, const void* rhs, void* result) {
            *static_cast<signed char*>(result) = key_strict_cmp(
                *static_cast<const Key*>(lhs),
                *static_cast<const Key*>(rhs));
        }
    ''')
    ltoir = core.compile_to_ltoir(source, prepend_cupy_headers=True)
    return ltoir, layout, source


def _run_merge_sort(compute, family, key, keys_in, keys_out, comparator,
                    num_items, *, values_in=None, values_out=None):
    """Build or reuse an algorithm and execute its two-call interface."""
    build_arguments = dict(
        d_in_keys=keys_in, d_out_keys=keys_out,
        d_in_values=values_in, d_out_values=values_out, op=comparator)
    # Build with real buffers: null proxy pointers are currently interpreted
    # as absent arrays. Addresses and array sizes are excluded from our key.
    sorter = cached_algorithm(
        family, key, None,
        lambda: compute.make_merge_sort(**build_arguments))
    execution_arguments = dict(
        build_arguments, num_items=num_items,
        stream=cupy.cuda.get_current_stream())
    tmp_size = sorter(temp_storage=None, **execution_arguments)
    workspace = cupy.empty(tmp_size, dtype=numpy.uint8)
    sorter(temp_storage=workspace, **execution_arguments)


def _copy_key_values(keys_out, result):
    """Extract values from packed keys without imposing vector alignment."""
    layout = keys_out.dtype
    value_ptr = keys_out.data + layout.fields['value'][1]
    if result.dtype.kind == 'c':
        # Packed complex fields have component alignment; copying as
        # real components avoids vector-alignment requirements.
        real_dtype = numpy.dtype(f'float{result.dtype.itemsize * 4}')
        values = cupy.ndarray(
            (result.size, 2), dtype=real_dtype, memptr=value_ptr,
            strides=(layout.itemsize, real_dtype.itemsize))
        result.view(real_dtype).reshape(-1, 2)[...] = values
    else:
        values = cupy.ndarray(
            (result.size,), dtype=result.dtype, memptr=value_ptr,
            strides=(layout.itemsize,))
        result.reshape(-1)[...] = values


def _sort(a, descending, *, return_indices):
    """Shared last-axis implementation for sort and argsort."""
    if (not isinstance(a, cupy.ndarray) or a.ndim == 0
            or not a.flags.c_contiguous
            or a.dtype.name not in _SUPPORTED_DTYPES):
        return None
    compute = _get_cuda_compute()
    if compute is None:
        return None
    with a.device:
        try:
            if a.size == 0 or a.shape[-1] <= 1:
                return (cupy.zeros(a.shape, dtype=numpy.intp)
                        if return_indices else a.copy())
            multi_row = a.ndim > 1
            ltoir, layout, source = _get_specialization(
                a.dtype, bool(descending), multi_row)
            comparator = compute.op.RawOp(ltoir=ltoir, name='strict_cmp')
            if multi_row:
                index_type = compute.types.from_numpy_dtype(_INDEX_DTYPE)
                width_state = numpy.intp(a.shape[-1]).tobytes()
                row_op = compute.op.RawOp(
                    ltoir=ltoir, name='row_id', state=width_state,
                    state_alignment=_INDEX_DTYPE.alignment)
                row_ids = compute.TransformIterator(
                    compute.CountingIterator(numpy.intp(0)), row_op,
                    value_type=index_type)
                keys_in = compute.ZipIterator(row_ids, a.reshape(-1))
                # Merge passes read output keys, so use packed storage with
                # the zip's layout instead of an output iterator proxy.
                assert keys_in.value_type.dtype.itemsize == layout.itemsize
                keys_out = _KeyBuffer(a.size, layout)
            else:
                keys_in = a
                keys_out = cupy.empty_like(a)
            if return_indices:
                result = cupy.empty(a.shape, dtype=numpy.intp)
                values_in = compute.CountingIterator(numpy.intp(0))
                if multi_row:
                    column_op = compute.op.RawOp(
                        ltoir=ltoir, name='column_id', state=width_state,
                        state_alignment=_INDEX_DTYPE.alignment)
                    values_in = compute.TransformIterator(
                        values_in, column_op, value_type=index_type)
                values_out = result.reshape(-1)
            else:
                result = cupy.empty_like(a) if multi_row else keys_out
                values_in = values_out = None
            operation = 'merge_argsort' if return_indices else 'merge_sort'
            family = operation + ('_nd' if multi_row else '_1d')
            key = (a.dtype, source)
            _run_merge_sort(
                compute, family, key, keys_in, keys_out, comparator, a.size,
                values_in=values_in, values_out=values_out)
            if multi_row and not return_indices:
                # Carry values only in the keys during sorting, then extract
                # once instead of moving a duplicate payload in every pass.
                _copy_key_values(keys_out, result)
        except memory.OutOfMemoryError:
            return None
        return result


def cuda_compute_merge_sort(a, descending=False):
    """Return a sorted copy of a C-contiguous CuPy array along its last axis.

    Return a new array on success, or None for unsupported inputs or an
    allocation failure, allowing the caller to select another backend. Uses
    the current stream on a's device. Stability relies on the current CUB
    implementation, as described in the module docstring. This is an internal
    accelerator entry point, not public dispatch.
    """
    return _sort(a, descending, return_indices=False)


def cuda_compute_argsort(a, descending=False):
    """Return stable last-axis indices, or None to request another backend."""
    return _sort(a, descending, return_indices=True)


@_util.memoize(for_each_device=True)
def _get_lexsort_specialization(dtype):
    source = _comparison_source(dtype.name, False) + dedent(r'''
        struct LexsortState {
            const T* keys;
            intptr_t num_keys;
            intptr_t row_width;
        };
        extern "C" __device__ void lex_strict_cmp(
                void* state, const void* lhs, const void* rhs, void* result) {
            const auto& s = *static_cast<const LexsortState*>(state);
            const auto i = *static_cast<const intptr_t*>(lhs);
            const auto j = *static_cast<const intptr_t*>(rhs);
            auto& out = *static_cast<signed char*>(result);
            // Last key is primary; earlier keys break ties in reverse order.
            for (intptr_t k = s.num_keys; k-- > 0;) {
                const T a = s.keys[k * s.row_width + i];
                const T b = s.keys[k * s.row_width + j];
                if (value_strict_cmp(a, b)) { out = 1; return; }
                if (value_strict_cmp(b, a)) { out = 0; return; }
            }
            out = 0;
        }
    ''')
    source += _layout_assertions('LexsortState', _LEXSORT_STATE_DTYPE)
    return core.compile_to_ltoir(source, prepend_cupy_headers=True), source


def cuda_compute_lexsort(keys):
    """Return stable indices for C-contiguous (k, n) keys, or None."""
    if (not isinstance(keys, cupy.ndarray) or keys.ndim != 2
            or not keys.flags.c_contiguous
            or keys.dtype.name not in _SUPPORTED_DTYPES):
        return None
    compute = _get_cuda_compute()
    if compute is None:
        return None
    with keys.device:
        try:
            num_keys, row_width = keys.shape
            if row_width <= 1 or num_keys == 0:
                return cupy.arange(row_width, dtype=numpy.intp)
            ltoir, source = _get_lexsort_specialization(keys.dtype)
            state = numpy.zeros((), dtype=_LEXSORT_STATE_DTYPE)
            state['keys'] = keys.data.ptr
            state['num_keys'] = num_keys
            state['row_width'] = row_width
            comparator = compute.op.RawOp(
                ltoir=ltoir, name='lex_strict_cmp',
                state=state.tobytes(),
                state_alignment=_LEXSORT_STATE_DTYPE.alignment)
            indices_in = compute.CountingIterator(numpy.intp(0))
            indices_out = cupy.empty(row_width, dtype=numpy.intp)
            key = (keys.dtype, source)
            _run_merge_sort(
                compute, 'merge_lexsort', key, indices_in, indices_out,
                comparator, row_width)
        except memory.OutOfMemoryError:
            return None
        return indices_out
