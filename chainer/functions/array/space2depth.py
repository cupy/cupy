from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy


class Space2Depth(function.Function):

    """Space to depth transformation."""

    def __init__(self, r):
        self.r = r

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == numpy.float32,
                          in_types[0].ndim == 4
                          )

    def forward(self, inputs):

        X, = inputs
        xp = cuda.get_array_module(X)
        bsize, c, a, b = X.shape
        X = xp.transpose(X, (0, 2, 3, 1))
        X = xp.reshape(X,
                       (bsize, a // self.r, self.r, b // self.r, self.r, c)
                       )
        X = xp.transpose(X, (0, 1, 3, 2, 4, 5))
        X = xp.reshape(X,
                       (bsize, a // self.r, b // self.r, self.r ** 2 * c)
                       )
        X = xp.transpose(X, (0, 3, 1, 2))
        return X,

    def backward(self, inputs, grad_outputs):
        gy, = grad_outputs
        xp = cuda.get_array_module(gy)
        bsize, c, a, b = gy.shape
        c //= self.r ** 2
        gy = xp.transpose(gy, (0, 2, 3, 1))
        gy = xp.reshape(gy, (bsize, a, b, self.r, self.r, c))
        gy = xp.transpose(gy, (0, 1, 3, 2, 4, 5))
        gy = xp.reshape(gy, (bsize, a * self.r, b * self.r, c))
        gy = xp.transpose(gy, (0, 3, 1, 2))
        return gy,


def space2depth(X, r):
    """Computes the space2depth transformation for subpixel calculations.

    Args:
        X (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable holding a 4d array of shape
            ``(batch, channel, dim1 * r, dim2 * r)``.
        r (int): the downscaling factor.

    Returns:
        ~chainer.Variable:
            A variable holding the downscaled layer array from subpixel array
            sampling. The shape is ``(batch, channel * r * r, dim1, dim2)``.

    .. note::
       This can be used to compute inverse super-resolution transformations.
       See https://arxiv.org/abs/1609.05158 for details.

    .. seealso:: :func:`depth2space`

    .. admonition:: Example

        >>> X = np.arange(24).reshape(1, 1, 4, 6).astype('f')
        >>> X.shape
        (1, 1, 4, 6)
        >>> X
        array([[[[  0.,   1.,   2.,   3.,   4.,   5.],
                 [  6.,   7.,   8.,   9.,  10.,  11.],
                 [ 12.,  13.,  14.,  15.,  16.,  17.],
                 [ 18.,  19.,  20.,  21.,  22.,  23.]]]], dtype=float32)
        >>> y = F.space2depth(X, 2)
        >>> y.shape
        (1, 4, 2, 3)
        >>> y.data
        array([[[[  0.,   2.,   4.],
                 [ 12.,  14.,  16.]],
        <BLANKLINE>
                [[  1.,   3.,   5.],
                 [ 13.,  15.,  17.]],
        <BLANKLINE>
                [[  6.,   8.,  10.],
                 [ 18.,  20.,  22.]],
        <BLANKLINE>
                [[  7.,   9.,  11.],
                 [ 19.,  21.,  23.]]]], dtype=float32)

    """
    return Space2Depth(r)(X)
