from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Copy(function.Function):

    """Copies the input variable onto the specified device."""

    def __init__(self, out_device):
        self.out_device = out_device

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1
        )

    def forward_cpu(self, x):
        if self.out_device == -1:
            return x[0].copy(),
        else:
            return cuda.to_gpu(x[0], device=self.out_device),

    def forward_gpu(self, x):
        if self.out_device == -1:
            return cuda.to_cpu(x[0]),
        else:
            return cuda.copy(x[0], out_device=self.out_device),

    def backward(self, inputs, grad_outputs):
        # In this function, `grad_outputs` contains cuda arrays even when
        # `inputs` only contains numpy arrays.
        if isinstance(inputs[0], cuda.ndarray):
            return self.backward_gpu(inputs, grad_outputs)
        else:
            return self.backward_cpu(inputs, grad_outputs)

    def backward_cpu(self, x, gy):
        if self.out_device == -1:
            return gy[0].copy(),
        else:
            return cuda.to_cpu(gy[0]),

    def backward_gpu(self, x, gy):
        if self.out_device == -1:
            return cuda.to_gpu(gy[0], device=cuda.get_device(x[0])),
        else:
            return cuda.copy(gy[0], out_device=cuda.get_device(x[0])),


def copy(x, dst):
    """Copies the input variable onto the specified device.

    This function copies the array of input variable onto the device specified
    by ``dst``. When ``dst == -1``, it copies the array onto the host memory.
    This function supports copies from host to host, from host to device,
    from device to device and from device to host.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable to be copied.
        dst (int): Target device specifier.

    Returns:
        ~chainer.Variable: Output variable.

    .. admonition:: Example

        >>> import chainer.cuda as cuda
        >>> x = np.random.uniform(-1, 1, (5, 10))
        >>> cuda.get_device_from_array(x).id
        -1
        >>> y = F.copy(x, 0) # from host to device0
        >>> cuda.get_device_from_array(y.data).id
        0
        >>> z = F.copy(y, -1) # from device0 to host
        >>> cuda.get_device_from_array(z.data).id
        -1

    """
    return Copy(dst)(x)
