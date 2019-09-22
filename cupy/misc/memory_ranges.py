def _get_bound(array):
    left = array.data.ptr
    right = left

    for dim, stride in zip(array.shape, array.strides):
        right += (dim - 1) * stride

    if left > right:
        left, right = right, left

    return left, right + array.itemsize


def _may_share_bounds(a, b):
    a_data, b_data = a.data, b.data

    if (a_data.device_id != b_data.device_id
            or a_data.mem != b_data.mem
            or a.size == 0 or b.size == 0):
        return False

    a_left, a_right = _get_bound(a)
    b_left, b_right = _get_bound(b)

    return a_left < b_right and b_left < a_right


def may_share_memory(a, b, max_work=None):
    if max_work is None:
        return _may_share_bounds(a, b)

    raise NotImplementedError('Only supported for `max_work` is `None`')


def shares_memory(a, b, max_work=None):
    if max_work == 'MAY_SHARE_BOUNDS':
        return _may_share_bounds(a, b)

    raise NotImplementedError(
        'Only supported for `max_work` is MAY_SHARE_BOUNDS')
