from __future__ import annotations

import numpy

import cupy
from cupy import _core
from cupy._core import _kernel
from cupy._core import _memory_range
from cupy._manipulation import join
from cupy._sorting import search


def may_share_memory(a, b, max_work=None):
    if max_work is None:
        return _memory_range.may_share_bounds(a, b)

    raise NotImplementedError('Only supported for `max_work` is `None`')


_get_memory_ptrs_kernel = _kernel.ElementwiseKernel(
    'T x', 'uint64 out',
    'out = (unsigned long long)(&x)',
    'cupy_get_memory_ptrs'
)


def _get_memory_ptrs_ascend(x):
    """ASCEND: ``&x`` is not expressible in an aclnn kernel
    (``cupy_get_memory_ptrs`` is not registered), so the per-element
    device addresses are computed on host from ``data.ptr`` plus the byte
    strides, then uploaded as uint64.
    """
    if x.size == 0:
        return cupy.array([], dtype=numpy.uint64)
    base = x.data.ptr
    flat = numpy.arange(x.size, dtype=numpy.int64)
    offsets = numpy.zeros(x.size, dtype=numpy.int64)
    for axis in range(x.ndim - 1, -1, -1):
        idx = flat % x.shape[axis]
        flat = flat // x.shape[axis]
        offsets += idx * x.strides[axis]
    return cupy.asarray((offsets + base).astype(numpy.uint64))


def _get_memory_ptrs(x):
    from cupy.backends.backend import is_ascend
    if is_ascend:
        # ASCEND: ElementwiseKernel cannot materialize device addresses,
        # compute per-element addresses on host instead
        if x.dtype.kind != 'c':
            return _get_memory_ptrs_ascend(x)
        return join.concatenate([
            _get_memory_ptrs_ascend(x.real),
            _get_memory_ptrs_ascend(x.imag)
        ])
    if x.dtype.kind != 'c':
        return _get_memory_ptrs_kernel(x)
    return join.concatenate([
        _get_memory_ptrs_kernel(x.real),
        _get_memory_ptrs_kernel(x.imag)
    ])


def shares_memory(a, b, max_work=None):
    if a is b and a.size != 0:
        return True
    if max_work == 'MAY_SHARE_BOUNDS':
        return _memory_range.may_share_bounds(a, b)

    if max_work in (None, 'MAY_SHARE_EXACT'):
        from cupy.backends.backend import is_ascend
        if is_ascend:
            # ASCEND: cupy_searchsorted_kernel is not registered either;
            # the ptr arrays are host-derived anyway, so finish the exact
            # match in numpy on host
            a_ptrs = _get_memory_ptrs(a).get().ravel()
            b_ptrs = _get_memory_ptrs(b).get().reshape(-1, 1)
            a_ptrs.sort()
            x = numpy.searchsorted(a_ptrs, b_ptrs, 'left')
            y = numpy.searchsorted(a_ptrs, b_ptrs, 'right')
            return bool((x != y).any())
        a_ptrs = _get_memory_ptrs(a).ravel()
        b_ptrs = _get_memory_ptrs(b).reshape(-1, 1)
        a_ptrs.sort()
        x = search.searchsorted(a_ptrs, b_ptrs, 'left')
        y = search.searchsorted(a_ptrs, b_ptrs, 'right')
        return bool((x != y).any())

    raise NotImplementedError('Not supported for integer `max_work`.')
