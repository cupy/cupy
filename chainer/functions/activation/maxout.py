from chainer.functions.array import reshape
from chainer.functions.math import minmax


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

    >>> l = L.Linear(in_size, out_size * pool_size)
    ... x = Variable(...)  # prepare data
    ... x = l(x)
    ... y = maxout(x, pool_size)

    Args:
       x (~chainer.Variable): Input variable. Its first dimension is assumed
            to be the *minibatch dimension*. The other dimensions are treated
            as one concatenated dimension.
    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`~chainer.links.Maxout`
    """

    msg = 'channel dimension must be divided by num_channel'
    assert x.data.shape[axis] % pool_size == 0, msg

    shape = (x.data.shape[:axis] +
             (x.data.shape[axis] // pool_size, pool_size) +
             x.data.shape[axis + 1:])
    x = reshape.reshape(x, shape)
    return minmax.max(x, axis=axis + 1)
