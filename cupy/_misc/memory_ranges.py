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
    'get_memory_ptrs'
)


def _get_memory_ptrs(x):
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
        a_ptrs = _get_memory_ptrs(a).ravel()
        b_ptrs = _get_memory_ptrs(b).reshape(-1, 1)
        a_ptrs.sort()
        x = search.searchsorted(a_ptrs, b_ptrs, 'left')
        y = search.searchsorted(a_ptrs, b_ptrs, 'right')
        return bool((x != y).any())

    raise NotImplementedError('Not supported for integer `max_work`.')
