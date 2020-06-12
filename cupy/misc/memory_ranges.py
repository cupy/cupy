from cupy.core import _kernel
from cupy.core import _memory_range


def may_share_memory(a, b, max_work=None):
    if max_work is None:
        return _memory_range.may_share_bounds(a, b)

    raise NotImplementedError('Only supported for `max_work` is `None`')


_get_memory_ptrs = _kernel.ElementwiseKernel(
    'T x', 'uint64 out',
    'out = (unsigned long long)(&x)',
    'get_memory_ptrs'
)


def shares_memory(a, b, max_work=None):
    if max_work == 'MAY_SHARE_BOUNDS':
        return _memory_range.may_share_bounds(a, b)

    if max_work in (None, 'MAY_SHARE_EXACT'):
        a_ptrs = _get_memory_ptrs(a).ravel()
        b_ptrs = _get_memory_ptrs(b).reshape(-1, 1)
        return bool((a_ptrs == b_ptrs).any())

    raise NotImplementedError('Not supported for integer `max_work`.')
