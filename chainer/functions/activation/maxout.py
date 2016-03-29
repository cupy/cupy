from chainer.functions.array import reshape
from chainer.functions.math import minmax
from chainer.utils import type_check


def maxout(x, pool_size, axis=1):
    """Maxout activation function.

    It accepts an input tensor ``x``, reshapes the ``axis`` dimension
    (say the size being ``M * pool_size``) into two dimensions
    ``(M, pool_size)``, and takes maximum along the ``axis`` dimension.
    The output of this function is same as ``x`` except that ``axis`` dimension
    is transformed from ``M * pool_size`` to ``M``.

    Typically, ``x`` is the output of a linear layer or a convolution layer.
    The following is the example where we use :func:`maxout` in combination
    with a Linear link.

    >>> import numpy, chainer, chainer.links as L
    >>> in_size, out_size, pool_size = 100, 100, 100
    >>> l = L.Linear(in_size, out_size * pool_size)
    >>> x = chainer.Variable(numpy.zeros((1, in_size), 'f'))  # prepare data
    >>> x = l(x)
    >>> y = maxout(x, pool_size)

    Args:
       x (~chainer.Variable): Input variable. Its first dimension is assumed
            to be the *minibatch dimension*. The other dimensions are treated
            as one concatenated dimension.
    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`~chainer.links.Maxout`
    """

    if pool_size <= 0:
        raise ValueError('pool_size must be a positive integer.')

    if x.data.shape[axis] % pool_size != 0:
        expect = 'x.data.shape[axis] % pool_size == 0'
        actual = 'x.data.shape[axis]={}, pool_size={}'.format(
            x.data.shape[axis], pool_size)
        msg = 'axis dimension must be divided by pool_size'
        raise type_check.InvalidType(expect, actual, msg)

    shape = (x.data.shape[:axis] +
             (x.data.shape[axis] // pool_size, pool_size) +
             x.data.shape[axis + 1:])
    x = reshape.reshape(x, shape)
    return minmax.max(x, axis=axis + 1)
