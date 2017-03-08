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
            x_type.dtype.kind == 'f',
            grid_type.dtype.kind == 'f',
            x_type.ndim == 4,
            grid_type.ndim == 4,
            grid_type.shape[1] == 2,
            x_type.shape[0] == grid_type.shape[0],
        )

    def forward_cpu(self, inputs):
        return self._forward(inputs)

    def forward_gpu(self, inputs):
        if cuda.cudnn_enabled and self.use_cudnn and _cudnn_version >= 5000:
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
        return self._forward(inputs)

    def _forward(self, inputs):
        x, grid = inputs
        xp = cuda.get_array_module(x)
        B, C, H, W = x.shape
        _, _, out_H, out_W = grid.shape

        grid = grid.reshape(grid.shape[:2] + (-1,))

        u = grid[:, 0]
        v = grid[:, 1]

        # clip coordinates to [-1, 1]
        u = u.clip(-1, 1)
        v = v.clip(-1, 1)

        # rescale coordiantes from [-1, 1] to [0, width or height - 1]
        u = (u + 1) / 2 * (W - 1)
        v = (v + 1) / 2 * (H - 1)

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

        ys = []
        for b in range(B):
            elem = (w1[b, :, None] * x[b, :, v0[b], u0[b]] +
                    w2[b, :, None] * x[b, :, v0[b], u1[b]] +
                    w3[b, :, None] * x[b, :, v1[b], u0[b]] +
                    w4[b, :, None] * x[b, :, v1[b], u1[b]])
            elem = elem.reshape(out_H, out_W, C)
            elem = elem.transpose(2, 0, 1)
            ys.append(elem)
        y = xp.concatenate([xp.expand_dims(y, axis=0) for y in ys], axis=0)
        return y,

    def backward_cpu(self, inputs, grad_outputs):
        return self._backward(inputs, grad_outputs)

    def backward_gpu(self, inputs, grad_outputs):
        if cuda.cudnn_enabled and self.use_cudnn and _cudnn_version >= 5000:
            x, grid = inputs
            gy, = grad_outputs

            grid_t = cuda.cupy.transpose(grid, (0, 2, 3, 1))
            gx = cuda.cupy.empty_like(x)
            ggrid_t = cuda.cupy.empty_like(grid_t)
            grid_t = cuda.cupy.ascontiguousarray(grid_t)

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
        return self._backward(inputs, grad_outputs)

    def _backward(self, inputs, grad_outputs):
        x, grid = inputs
        xp = cuda.get_array_module(x)
        gy, = grad_outputs

        B, C, H, W = x.shape
        _, _, out_H, out_W = grid.shape
        grid = grid.reshape(grid.shape[:2] + (-1,))

        u = grid[:, 0]
        v = grid[:, 1]

        # clip coordinates to [-1, 1]
        u = u.clip(-1, 1)
        v = v.clip(-1, 1)

        # rescale coordiantes from [-1, 1] to [0, width or height - 1]
        u = (u + 1) / 2. * (W - 1)
        v = (v + 1) / 2. * (H - 1)

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

        # --- gu, gv
        gus = []
        gvs = []
        for b in range(B):
            gu_elem = (-wv1[b, :, None] * x[b, :, v0[b], u0[b]] +
                       wv1[b, :, None] * x[b, :, v0[b], u1[b]] -
                       wv0[b, :, None] * x[b, :, v1[b], u0[b]] +
                       wv0[b, :, None] * x[b, :, v1[b], u1[b]])
            gu_elem = gu_elem.reshape(out_H, out_W, C)
            gu_elem = gu_elem.transpose(2, 0, 1)
            gus.append(gu_elem)
            gv_elem = (-wu1[b, :, None] * x[b, :, v0[b], u0[b]] -
                       wu0[b, :, None] * x[b, :, v0[b], u1[b]] +
                       wu1[b, :, None] * x[b, :, v1[b], u0[b]] +
                       wu0[b, :, None] * x[b, :, v1[b], u1[b]])
            gv_elem = gv_elem.reshape(out_H, out_W, C)
            gv_elem = gv_elem.transpose(2, 0, 1)
            gvs.append(gv_elem)
        gu = xp.concatenate([xp.expand_dims(g, axis=0) for g in gus], axis=0)
        gv = xp.concatenate([xp.expand_dims(g, axis=0) for g in gvs], axis=0)

        gu *= gy
        gv *= gy
        gu = xp.sum(gu, axis=1)
        gv = xp.sum(gv, axis=1)
        # this offsets scaling of the coordinates
        gu = gu / 2. * (W - 1)
        gv = gv / 2. * (H - 1)

        ggrid = xp.concatenate((gu[:, None], gv[:, None]), axis=1)

        # --- gx
        if xp is numpy:
            scatter_add = numpy.add.at
        else:
            scatter_add = xp.scatter_add
        gxs = []
        gy = gy.reshape(B, C, -1)
        for b in range(B):
            gx = xp.zeros_like(x[b])
            scatter_add(gx, (slice(None), v0[b], u0[b]),
                        gy[b] * wu1[b] * wv1[b])
            scatter_add(gx, (slice(None), v0[b], u1[b]),
                        gy[b] * wu0[b] * wv1[b])
            scatter_add(gx, (slice(None), v1[b], u0[b]),
                        gy[b] * wu1[b] * wv0[b])
            scatter_add(gx, (slice(None), v1[b], u1[b]),
                        gy[b] * wu0[b] * wv0[b])
            gxs.append(gx)
        gx = xp.concatenate([xp.expand_dims(g, axis=0) for g in gxs], axis=0)
        return gx, ggrid


def spatial_transformer_sampler(x, grid, use_cudnn=True):
    """2D Spatial Transformer sampler.

    This is a differentiable image sampler. With a set of sampling points
    ``grid`` and an input feature map ``x``, this produces a sampled output
    feature map.

    This function currently only supports bilinear interpolation as a sampling
    kernel.

    It is important to note that this function assumes values in ``grid`` to
    be in range :math:`[-1, 1]`.

    Notatition: here is a notation for dimensionalities.

    - :math:`n` is the batch size.
    - :math:`c_I` is the number of the input channels.
    - :math:`h` and :math:`w` are the height and width of the input image,
      respectively.
    - :math:`h_O` and :math:`w_O` are the height and width of the output
      image.

    See detail in a paper: `Spatial Transformer Networks \
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

            In the 2nd dimension, the first coordinate corresponds to the
            location along the horizontal axis, and the second coordinate
            corresponds to the location along the vertical axis.

            The values of this variable is clipped in range :math:`[-1, 1]`.
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
