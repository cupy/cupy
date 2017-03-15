import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    _algorithm = libcudnn.CUDNN_SOFTMAX_LOG
    _mode = libcudnn.CUDNN_SOFTMAX_MODE_CHANNEL


def logsumexp(x):
    xp = cuda.get_array_module(x)
    m = x.max(axis=1, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    s = y.sum(axis=1, keepdims=True)
    xp.log(s, out=s)
    m += s
    return m


def _log_softmax(x, use_cudnn):
    if cuda.cudnn_enabled and use_cudnn and _cudnn_version >= 3000:
        xp = cuda.get_array_module(x)
        if xp != numpy:
            oz_dtype = 'd' if x.dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes
            handle = cudnn.get_handle()
            x_cube = x.reshape(x.shape[:2] + (-1, 1))
            desc = cudnn.create_tensor_descriptor(x_cube)
            y = xp.empty_like(x)
            libcudnn.softmaxForward(
                handle, _algorithm, _mode, one.data, desc.value,
                x_cube.data.ptr, zero.data, desc.value,
                y.data.ptr)
            return y
    log_z = logsumexp(x)
    y = x - log_z
    return y


class LogSoftmax(function.Function):

    """Log-softmax activation function."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn
        self.y = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim > 1,
        )

    def forward(self, xs):
        self.y = _log_softmax(xs[0], self.use_cudnn)
        return self.y,

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        if (xp != numpy and cuda.cudnn_enabled and self.use_cudnn and
                _cudnn_version >= 3000):
            oz_dtype = 'd' if x[0].dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes
            handle = cudnn.get_handle()
            gx = xp.empty_like(x[0])
            gx_cube = gx.reshape(gx.shape[:2] + (-1, 1))
            desc = cudnn.create_tensor_descriptor(gx_cube)
            libcudnn.softmaxBackward(
                handle, _algorithm, _mode, one.data, desc.value,
                self.y.data.ptr, desc.value, gy[0].data.ptr, zero.data,
                desc.value, gx.data.ptr)
        else:
            gx = gy[0] - xp.exp(self.y) * gy[0].sum(axis=1, keepdims=True)

        return gx,


def log_softmax(x, use_cudnn=True):
    """Channel-wise log-softmax function.

    This function computes its logarithm of softmax along the second axis. Let
    :math:`x = (x_1, x_2, \\dots, x_D)^{\\top}` be the D dimensional input
    array.
    For each input array :math:`x`, it computes the logarithm of the function
    :math:`\log f(x)` defined as

    .. math::
        f(x) = {\\exp(x) \\over \\sum_{x_d} \\exp(x_d)}.

    This method is theoretically equivalent to ``log(softmax(x))`` but is more
    stable.

    .. note::
        ``log(softmax(x))`` may cause underflow when ``x`` is too small,
        because ``softmax(x)`` may returns ``0``.
        ``log_softmax`` method is more stable.

    Args:
        X (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable.
            A :math:`n`-dimensional (:math:`n \\geq 2`) float array.
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.
        A :math:`n`-dimensional (:math:`n \\geq 2`) float array, which is the
        same shape with X.

    .. seealso:: :func:`~chainer.functions.softmax`

    .. admonition:: Example

        >>> x = np.array([[0, 1, 2], [0, 2, 4]], 'f')
        >>> x
        array([[ 0.,  1.,  2.],
               [ 0.,  2.,  4.]], dtype=float32)
        >>> F.log_softmax(x).data
        array([[-2.40760589, -1.40760589, -0.40760589],
               [-4.14293146, -2.14293146, -0.14293146]], dtype=float32)

    """
    return LogSoftmax(use_cudnn)(x)
