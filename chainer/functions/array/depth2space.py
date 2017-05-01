from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy


class Depth2Space(function.Function):

    """Depth to space transformation."""

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
        c //= self.r ** 2
        X = xp.transpose(X, (0, 2, 3, 1))
        X = xp.reshape(X, (bsize, a, b, self.r, self.r, c))
        X = xp.transpose(X, (0, 1, 3, 2, 4, 5))
        X = xp.reshape(X, (bsize, a * self.r, b * self.r, c))
        X = xp.transpose(X, (0, 3, 1, 2))
        return X,

    def backward(self, inputs, grad_outputs):
        gy, = grad_outputs
        xp = cuda.get_array_module(gy)
        bsize, c, a, b = gy.shape
        gy = xp.transpose(gy, (0, 2, 3, 1))
        gy = xp.reshape(gy,
                        (bsize, a // self.r, self.r, b // self.r, self.r, c)
                        )
        gy = xp.transpose(gy, (0, 1, 3, 2, 4, 5))
        gy = xp.reshape(gy,
                        (bsize, a // self.r, b // self.r, self.r ** 2 * c)
                        )
        gy = xp.transpose(gy, (0, 3, 1, 2))
        return gy,


def depth2space(X, r):
    """Computes the depth2space transformation for subpixel calculations.

    Args:
        X (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable holding a 4d array of shape
            ``(batch, channel * r * r, dim1, dim2)``.
        r (int): the upscaling factor.

    Returns:
        ~chainer.Variable:
            A variable holding the upscaled array from
            interspersed depth layers. The shape is
            ``(batch, channel, dim1 * r, dim2 * r)``.

    .. note::
       This can be used to compute super-resolution transformations.
       See https://arxiv.org/abs/1609.05158 for details.

    .. seealso:: :func:`space2depth`

    .. admonition:: Example

        >>> X = np.arange(24).reshape(1, 4, 2, 3).astype('f')
        >>> X.shape
        (1, 4, 2, 3)
        >>> X
        array([[[[  0.,   1.,   2.],
                 [  3.,   4.,   5.]],
        <BLANKLINE>
                [[  6.,   7.,   8.],
                 [  9.,  10.,  11.]],
        <BLANKLINE>
                [[ 12.,  13.,  14.],
                 [ 15.,  16.,  17.]],
        <BLANKLINE>
                [[ 18.,  19.,  20.],
                 [ 21.,  22.,  23.]]]], dtype=float32)
        >>> y = F.depth2space(X, 2)
        >>> y.shape
        (1, 1, 4, 6)
        >>> y.data
        array([[[[  0.,   6.,   1.,   7.,   2.,   8.],
                 [ 12.,  18.,  13.,  19.,  14.,  20.],
                 [  3.,   9.,   4.,  10.,   5.,  11.],
                 [ 15.,  21.,  16.,  22.,  17.,  23.]]]], dtype=float32)

    """
    return Depth2Space(r)(X)
