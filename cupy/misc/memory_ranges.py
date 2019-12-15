from cupy.core import _memory_range


def may_share_memory(a, b, max_work=None):
    if max_work is None:
        return _memory_range.may_share_bounds(a, b)

    raise NotImplementedError('Only supported for `max_work` is `None`')


def shares_memory(a, b, max_work=None):
    if max_work == 'MAY_SHARE_BOUNDS':
        return _memory_range.may_share_bounds(a, b)

    raise NotImplementedError(
        'Only supported for `max_work` is MAY_SHARE_BOUNDS')
