import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    _sampler_type = libcudnn.CUDNN_SAMPLER_BILINEAR


class SpatialTransformerSampler(function.Function):

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 == n_in)

        x_type = in_types[0]
        grid_type = in_types[1]
        type_check.expect(
            x_type.dtype.char == 'f',
            grid_type.dtype.char == 'f',
            x_type.ndim == 4,
            grid_type.ndim == 4,
            grid_type.shape[1] == 2,
            x_type.shape[0] == grid_type.shape[0],
        )

    def forward_cpu(self, inputs):
        return self._forward(inputs)

    def forward_gpu(self, inputs):
        if not (cuda.cudnn_enabled and self.use_cudnn and
                _cudnn_version >= 5000):
            return self._forward(inputs)
        x, grid = inputs
        out_shape = x.shape[:2] + grid.shape[2:]
        y = cuda.cupy.empty(out_shape, dtype=x.dtype)
        shape = numpy.array(out_shape, dtype=numpy.int32)
        x = cuda.cupy.ascontiguousarray(x)
        grid_t = cuda.cupy.transpose(grid, (0, 2, 3, 1))
        grid_t = cuda.cupy.ascontiguousarray(grid_t)

        handle = cudnn.get_handle()
        x_desc = cudnn.create_tensor_descriptor(x)
        y_desc = cudnn.create_tensor_descriptor(y)

        self.st_desc =\
            cuda.cupy.cudnn.create_spatial_transformer_descriptor(
                _sampler_type, grid.dtype, len(shape), shape.ctypes.data)

        one = numpy.array(1, dtype=x.dtype).ctypes
        zero = numpy.array(0, dtype=x.dtype).ctypes
        libcudnn.spatialTfSamplerForward(
            handle, self.st_desc.value, one.data,
            x_desc.value, x.data.ptr, grid_t.data.ptr, zero.data,
            y_desc.value, y.data.ptr)
        return y,

    def _forward(self, inputs):
        x, grid = inputs
        xp = cuda.get_array_module(x)
        B, C, H, W = x.shape
        _, _, out_H, out_W = grid.shape

        grid = grid.reshape(grid.shape[:2] + (-1,))

        u = grid[:, 0]
        v = grid[:, 1]

        # Pad the image so that pixels locating outside of the original
        # image's size can be sampled.
        x_pad = xp.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')

        # Rescale coordinates from [-1, 1] to [0, width or height - 1],
        # and adjust them to the padded image.
        u = (u + 1) * (W - 1) / 2 + 1
        v = (v + 1) * (H - 1) / 2 + 1

        u_clipped = u.clip(0, W + 1)
        v_clipped = v.clip(0, H + 1)

        # indices of the 2x2 pixel neighborhood surrounding the coordinates
        u0 = xp.floor(u_clipped).astype(numpy.int32)
        u0 = u0.clip(0, W)
        u1 = u0 + 1
        v0 = xp.floor(v_clipped).astype(numpy.int32)
        v0 = v0.clip(0, H)
        v1 = v0 + 1

        # weights
        w1 = (u1 - u_clipped) * (v1 - v_clipped)
        w2 = (u_clipped - u0) * (v1 - v_clipped)
        w3 = (u1 - u_clipped) * (v_clipped - v0)
        w4 = (u_clipped - u0) * (v_clipped - v0)
        w1 = w1.astype(x_pad.dtype)
        w2 = w2.astype(x_pad.dtype)
        w3 = w3.astype(x_pad.dtype)
        w4 = w4.astype(x_pad.dtype)

        x_indexed_1 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, v0[b], u0[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_2 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, v0[b], u1[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_3 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, v1[b], u0[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_4 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, v1[b], u1[b]], axis=0) for b in range(B)], axis=0)
        y = w1[:, :, None] * x_indexed_1
        y += w2[:, :, None] * x_indexed_2
        y += w3[:, :, None] * x_indexed_3
        y += w4[:, :, None] * x_indexed_4

        y = y.reshape(B, out_H, out_W, C).transpose(0, 3, 1, 2)
        return y,

    def backward_cpu(self, inputs, grad_outputs):
        return self._backward(inputs, grad_outputs)

    def backward_gpu(self, inputs, grad_outputs):
        if not (cuda.cudnn_enabled and self.use_cudnn and
                _cudnn_version >= 5000):
            return self._backward(inputs, grad_outputs)
        x, grid = inputs
        gy, = grad_outputs

        grid_t = cuda.cupy.transpose(grid, (0, 2, 3, 1))
        grid_t = cuda.cupy.ascontiguousarray(grid_t)
        x = cuda.cupy.ascontiguousarray(x)
        gy = cuda.cupy.ascontiguousarray(gy)
        gx = cuda.cupy.empty_like(x)
        ggrid_t = cuda.cupy.empty_like(grid_t)

        handle = cudnn.get_handle()
        x_desc = cudnn.create_tensor_descriptor(x)
        dx_desc = cudnn.create_tensor_descriptor(gx)
        dy_desc = cudnn.create_tensor_descriptor(gy)

        one = numpy.array(1, dtype=x.dtype).ctypes
        zero = numpy.array(0, dtype=x.dtype).ctypes
        libcudnn.spatialTfSamplerBackward(
            handle, self.st_desc.value,
            one.data,
            x_desc.value, x.data.ptr,
            zero.data,
            dx_desc.value, gx.data.ptr,
            one.data,
            dy_desc.value, gy.data.ptr,
            grid_t.data.ptr, zero.data, ggrid_t.data.ptr)
        ggrid = cuda.cupy.transpose(ggrid_t, axes=(0, 3, 1, 2))
        return gx, ggrid

    def _backward(self, inputs, grad_outputs):
        x, grid = inputs
        xp = cuda.get_array_module(x)
        gy, = grad_outputs

        B, C, H, W = x.shape
        _, _, out_H, out_W = grid.shape
        grid = grid.reshape(grid.shape[:2] + (-1,))

        u = grid[:, 0]
        v = grid[:, 1]

        # Pad the image so that points locating outside of the original
        # image's size can be sampled.
        x_pad = xp.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')

        # Rescale coordinates from [-1, 1] to [0, width or height - 1],
        # and adjust them to the padded image.
        u = (u + 1) * (W - 1) / 2 + 1
        v = (v + 1) * (H - 1) / 2 + 1

        u_clipped = u.clip(0, W + 1)
        v_clipped = v.clip(0, H + 1)

        # indices of the 2x2 pixel neighborhood surrounding the coordinates
        u0 = xp.floor(u_clipped).astype(numpy.int32)
        u0 = u0.clip(0, W)
        u1 = u0 + 1
        v0 = xp.floor(v_clipped).astype(numpy.int32)
        v0 = v0.clip(0, H)
        v1 = v0 + 1

        # weights
        wu0 = u_clipped - u0
        wu1 = u1 - u_clipped
        wv0 = v_clipped - v0
        wv1 = v1 - v_clipped
        wu0 = wu0.astype(gy.dtype)
        wu1 = wu1.astype(gy.dtype)
        wv0 = wv0.astype(gy.dtype)
        wv1 = wv1.astype(gy.dtype)

        # --- gu, gv
        x_indexed_1 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, v0[b], u0[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_2 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, v0[b], u1[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_3 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, v1[b], u0[b]], axis=0) for b in range(B)], axis=0)
        x_indexed_4 = xp.concatenate([xp.expand_dims(
            x_pad[b, :, v1[b], u1[b]], axis=0) for b in range(B)], axis=0)

        gu = -wv1[:, :, None] * x_indexed_1
        gu += wv1[:, :, None] * x_indexed_2
        gu -= wv0[:, :, None] * x_indexed_3
        gu += wv0[:, :, None] * x_indexed_4

        gv = -wu1[:, :, None] * x_indexed_1
        gv -= wu0[:, :, None] * x_indexed_2
        gv += wu1[:, :, None] * x_indexed_3
        gv += wu0[:, :, None] * x_indexed_4

        gu = gu.reshape(B, out_H, out_W, C).transpose(0, 3, 1, 2)
        gv = gv.reshape(B, out_H, out_W, C).transpose(0, 3, 1, 2)

        gu *= gy
        gv *= gy
        gu = xp.sum(gu, axis=1)
        gv = xp.sum(gv, axis=1)
        # Offsets scaling of the coordinates and clip gradients.
        u_reshaped = u.reshape(gu.shape)
        v_reshaped = v.reshape(gv.shape)
        gu = gu / 2. * (W - 1) * (u_reshaped > 0) * (u_reshaped < (W + 1))
        gv = gv / 2. * (H - 1) * (v_reshaped > 0) * (v_reshaped < (H + 1))

        ggrid = xp.concatenate((gu[:, None], gv[:, None]), axis=1)

        # --- gx
        if xp is numpy:
            scatter_add = numpy.add.at
        else:
            scatter_add = xp.scatter_add
        gx = xp.zeros_like(x_pad)
        gy = gy.reshape(B, C, -1)
        for b in range(B):
            scatter_add(gx[b], (slice(None), v0[b], u0[b]),
                        gy[b] * wu1[b] * wv1[b])
            scatter_add(gx[b], (slice(None), v0[b], u1[b]),
                        gy[b] * wu0[b] * wv1[b])
            scatter_add(gx[b], (slice(None), v1[b], u0[b]),
                        gy[b] * wu1[b] * wv0[b])
            scatter_add(gx[b], (slice(None), v1[b], u1[b]),
                        gy[b] * wu0[b] * wv0[b])
        gx = gx[:, :, 1:-1, 1:-1]
        return gx, ggrid


def spatial_transformer_sampler(x, grid, use_cudnn=True):
    """2D Spatial Transformer sampler.

    This is a differentiable image sampler. With a set of sampling points
    ``grid`` and an input feature map ``x``, this produces a sampled output
    feature map.

    This function currently only supports bilinear interpolation as a sampling
    kernel.

    When coordinates in ``grid`` is outside range :math:`[-1, 1]`, values are
    sampled from a zero padded input image.

    Notatition: here is a notation for dimensionalities.

    - :math:`n` is the batch size.
    - :math:`c_I` is the number of the input channels.
    - :math:`h` and :math:`w` are the height and width of the input image,
      respectively.
    - :math:`h_O` and :math:`w_O` are the height and width of the output
      image.

    See detail in the following paper: `Spatial Transformer Networks \
    <https://arxiv.org/abs/1506.02025>`_.

    Args:
        x (~chainer.Variable):  Input variable of shape :math:`(n, c_I, h, w)`.
        grid (~chainer.Variable): Coordinate variable of shape
            :math:`(n, 2, h_O, w_O)`. Each coordinate defines the spatial
            location in the input where a sampling kernel is applied to get
            the value at a particular pixel in the output.
            ``grid[idx, :, i, j]`` corresponds to the coordinate that is used
            to sample the values for an output pixel at location
            :math:`(i, j)`.

            In the second dimension, the first coordinate corresponds to the
            location along the horizontal axis, and the second coordinate
            corresponds to the location along the vertical axis.

            The coordinate :math:`(-1, -1)` corresponds to the upper-left
            corner of the input image.
        use_cudnn (bool): If ``True``, then this function uses cuDNN if
            available. Note that, cuDNN supports SpatialTransformerSampler
            from version 5.0.0.

    Returns:
        ~chainer.Variable: Output feature map of shape \
            :math:`(n, c_I, h_O, w_O)`.

    """
    return SpatialTransformerSampler(use_cudnn)(x, grid)
