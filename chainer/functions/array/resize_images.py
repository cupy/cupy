import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class ResizeImages(function.Function):

    def __init__(self, output_shape):
        self.out_H = output_shape[0]
        self.out_W = output_shape[1]

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(n_in == 1)

        x_type = in_types[0]
        type_check.expect(
            x_type.dtype.char == 'f',
            x_type.ndim == 4
        )

    def forward(self, inputs):
        x, = inputs
        xp = cuda.get_array_module(x)

        B, C, H, W = x.shape

        u_1d = xp.linspace(0, W - 1, num=self.out_W)
        v_1d = xp.linspace(0, H - 1, num=self.out_H)
        grid = xp.meshgrid(u_1d, v_1d)
        # u, v are of shape (out_H * out_W,)
        u = grid[0].ravel()
        v = grid[1].ravel()

        # indices of the 2x2 pixel neighborhood surrounding the coordinates
        u0 = xp.floor(u).astype(numpy.int32)
        u0 = u0.clip(0, W - 2)
        u1 = u0 + 1
        v0 = xp.floor(v).astype(numpy.int32)
        v0 = v0.clip(0, H - 2)
        v1 = v0 + 1

        # weights
        w1 = (u1 - u) * (v1 - v)
        w2 = (u - u0) * (v1 - v)
        w3 = (u1 - u) * (v - v0)
        w4 = (u - u0) * (v - v0)
        w1 = w1.astype(x.dtype)
        w2 = w2.astype(x.dtype)
        w3 = w3.astype(x.dtype)
        w4 = w4.astype(x.dtype)

        y = (w1[None, None, :] * x[:, :, v0, u0] +
             w2[None, None, :] * x[:, :, v0, u1] +
             w3[None, None, :] * x[:, :, v1, u0] +
             w4[None, None, :] * x[:, :, v1, u1])
        y = y.reshape(B, C, self.out_H, self.out_W)
        return y,

    def backward(self, inputs, grad_outputs):
        x, = inputs
        xp = cuda.get_array_module(x)
        gy, = grad_outputs

        B, C, H, W = x.shape

        u_1d = xp.linspace(0, W - 1, num=self.out_W)
        v_1d = xp.linspace(0, H - 1, num=self.out_H)
        grid = xp.meshgrid(u_1d, v_1d)
        # u, v are of shape (out_H * out_W,)
        u = grid[0].ravel()
        v = grid[1].ravel()

        # indices of the 2x2 pixel neighborhood surrounding the coordinates
        u0 = xp.floor(u).astype(numpy.int32)
        u0 = u0.clip(0, W - 2)
        u1 = u0 + 1
        v0 = xp.floor(v).astype(numpy.int32)
        v0 = v0.clip(0, H - 2)
        v1 = v0 + 1

        # weights
        wu0 = u - u0
        wu1 = u1 - u
        wv0 = v - v0
        wv1 = v1 - v
        wu0 = wu0.astype(gy.dtype)
        wu1 = wu1.astype(gy.dtype)
        wv0 = wv0.astype(gy.dtype)
        wv1 = wv1.astype(gy.dtype)

        # --- gx
        if xp is numpy:
            scatter_add = numpy.add.at
        else:
            scatter_add = xp.scatter_add

        gx = xp.zeros_like(x)
        gy = gy.reshape(B, C, -1)
        scatter_add(gx, (slice(None), slice(None), v0, u0), gy * wu1 * wv1)
        scatter_add(gx, (slice(None), slice(None), v0, u1), gy * wu0 * wv1)
        scatter_add(gx, (slice(None), slice(None), v1, u0), gy * wu1 * wv0)
        scatter_add(gx, (slice(None), slice(None), v1, u1), gy * wu0 * wv0)
        return gx,


def resize_images(x, output_shape):
    """Resize images to the given shape.

    This function resizes 2D data to :obj:`output_shape`.
    Currently, only bilinear interpolation is supported as the sampling method.

    Notatition: here is a notation for dimensionalities.

    - :math:`n` is the batch size.
    - :math:`c_I` is the number of the input channels.
    - :math:`h` and :math:`w` are the height and width of the input image,
      respectively.
    - :math:`h_O` and :math:`w_O` are the height and width of the output
      image.

    Args:
        x (~chainer.Variable):  Input variable of shape :math:`(n, c_I, h, w)`.
        output_shape (tuple): This is a tuple of length 2 whose values are
            :obj:`(h_O, w_O)`. Note that the order of height and width is
            opposite of the one in OpenCV.

    Returns:
        ~chainer.Variable: Resized image whose shape is \
            :math:`(n, c_I, h_O, w_O)`.

    """
    return ResizeImages(output_shape)(x)
