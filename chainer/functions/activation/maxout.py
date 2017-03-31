from chainer.functions.array import reshape
from chainer.functions.math import minmax
from chainer.utils import type_check


def maxout(x, pool_size, axis=1):
    """Maxout activation function.

    It accepts an input tensor ``x``, reshapes the ``axis`` dimension
    (say the size being ``M * pool_size``) into two dimensions
    ``(M, pool_size)``, and takes maximum along the ``axis`` dimension.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`n`-dimensional (:math:`n \\ge` ``axis``)
            float array. In general, its first dimension is assumed to be the
            *minibatch dimension*. The other dimensions are treated as one
            concatenated dimension.
        pool_size (int):
            The size used for downsampling of pooling layer.
        axis (int):
            The ``axis`` dimension to be reshaped. The size of ``axis``
            dimension should be ``M * pool_size``.

    Returns:
        ~chainer.Variable:
            Output variable. The shape of the output is same as ``x`` except
            that ``axis`` dimension is transformed from ``M * pool_size`` to
            ``M``.

    .. seealso:: :class:`~chainer.links.Maxout`

    .. admonition:: Example

        Typically, ``x`` is the output of a linear layer or a convolution
        layer. The following is the example where we use :func:`maxout` in
        combination with a Linear link.

        >>> in_size, out_size, pool_size = 10, 10, 10
        >>> bias = np.arange(out_size * pool_size).astype('f')
        >>> l = L.Linear(in_size, out_size * pool_size, initial_bias=bias)
        >>> x = np.zeros((1, in_size), 'f')  # prepare data
        >>> x = l(x)
        >>> y = F.maxout(x, pool_size)
        >>> x.shape
        (1, 100)
        >>> y.shape
        (1, 10)
        >>> x.reshape((out_size, pool_size)).data
        array([[  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.],
               [ 10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.],
               [ 20.,  21.,  22.,  23.,  24.,  25.,  26.,  27.,  28.,  29.],
               [ 30.,  31.,  32.,  33.,  34.,  35.,  36.,  37.,  38.,  39.],
               [ 40.,  41.,  42.,  43.,  44.,  45.,  46.,  47.,  48.,  49.],
               [ 50.,  51.,  52.,  53.,  54.,  55.,  56.,  57.,  58.,  59.],
               [ 60.,  61.,  62.,  63.,  64.,  65.,  66.,  67.,  68.,  69.],
               [ 70.,  71.,  72.,  73.,  74.,  75.,  76.,  77.,  78.,  79.],
               [ 80.,  81.,  82.,  83.,  84.,  85.,  86.,  87.,  88.,  89.],
               [ 90.,  91.,  92.,  93.,  94.,  95.,  96.,  97.,  98.,  99.]], \
dtype=float32)
        >>> y.data
        array([[  9.,  19.,  29.,  39.,  49.,  59.,  69.,  79.,  89.,  99.]], \
dtype=float32)

    """

    if pool_size <= 0:
        raise ValueError('pool_size must be a positive integer.')

    x_shape = x.shape
    if x_shape[axis] % pool_size != 0:
        expect = 'x.shape[axis] % pool_size == 0'
        actual = 'x.shape[axis]={}, pool_size={}'.format(
            x_shape[axis], pool_size)
        msg = 'axis dimension must be divided by pool_size'
        raise type_check.InvalidType(expect, actual, msg)

    shape = (x_shape[:axis] +
             (x_shape[axis] // pool_size, pool_size) +
             x_shape[axis + 1:])
    x = reshape.reshape(x, shape)
    return minmax.max(x, axis=axis + 1)
