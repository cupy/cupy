import cupy


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
