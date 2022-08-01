import cupy


# Note: here, we used three kernel switches for the
# implementation, we tried using `cupy.fusion`,
# but the performance degraded. Because CuPy fusion does
# not generate competitive codes for all cases.
# The fusion technique we iterated:
#
# def make_expander(shape, axis):
#     axis = internal._normalize_axis_indices(axis, len(shape))
#     expander = []
#     for i, s in enumerate(x.shape):
#         if i in axis:
#             expander.append(None)
#         else:
#             expander.append(slice(None))
#     return tuple(expander)
#
# @_util.memoize(for_each_device=True)
# def _softmax_fuse(shape, axis):
#     expander = make_expander(shape, axis)
#     @_core.fusion.fuse()
#     def softmax_fuse(x):
#         x_max = cupy.amax(x, axis=axis)
#         exp_x_shifted = cupy.exp(x - x_max[expander])
#         return exp_x_shifted / cupy.sum(exp_x_shifted, axis=axis)[expander]
#     return softmax_fuse
#
# def softmax(x, axis=None):
#     fused_softmax = _softmax_fuse(shape=x.shape, axis=axis)
#     return fused_softmax(x)
#
# TODO: Add the `shape` method in cupy.fusion so we can
# fuse the make_expander function, too, which might
# increment the performance.


def softmax(x, axis=None):
    """Softmax function.

    The softmax function transforms each element of a
    collection by computing the exponential of each element
    divided by the sum of the exponentials of all the elements.

    Parameters
    ----------
    x : array-like
        The input array
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None

    Returns
    -------
    s : cupy.ndarray
        Returns an array with same shape as input. The result
        will sum to 1 along the provided axis

    """

    x_max = cupy.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = cupy.exp(x - x_max)
    return exp_x_shifted / cupy.sum(exp_x_shifted, axis=axis, keepdims=True)
