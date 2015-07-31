"""Device, context and memory management on CuPy.

Chainer uses CuPy (with very thin wrapper) to exploit the speed of GPU
computation. Following modules and classes are imported to :mod:`cuda`
module for convenience (refer to this table when reading chainer's source
codes).

============================ =================================
 imported name                original name
============================ =================================
 ``chainer.cuda.cupy``        :mod:`cupy`
 ``chainer.cuda.ndarray``     :mod:`cupy.ndarray`
 ``chainer.cuda.cupy.cuda``   :mod:`cupy.cuda`
 ``chainer.cuda.Device``      :mod:`cupy.cuda.Device`
 ``chainer.cuda.Event``       :mod:`cupy.cuda.Event`
 ``chainer.cuda.Stream``      :mod:`cupy.cuda.Stream`
============================ =================================

Chainer replaces the default allocator of CuPy by its memory pool
implementation. It enables us to reuse the device memory over multiple
forward/backward computations, and temporary arrays for consecutive elementwise
operations.
"""

import os

import numpy

_requires = []
try:
    import cupy
    import cupy.cuda
    import cupy.cuda.cublas
    import cupy.cudnn
    import cupy.random

    available = True
    cublas = cupy.cuda.cublas
    cudnn = cupy.cudnn
    random = cupy.random

    cudnn_enabled = int(os.environ.get('CHAINER_CUDNN', '1')) != 0
except Exception as e:
    available = False
    cudnn_enabled = False
    _resolution_error = e

# ------------------------------------------------------------------------------
# Basic types
# ------------------------------------------------------------------------------
if available:
    from cupy import cuda
    Device = cuda.Device
    Event = cuda.Event
    Stream = cuda.Stream
    ndarray = cupy.ndarray
else:
    # Dummy classes
    class Device(object):

        def use(self):
            pass

    class Event(object):
        pass

    class Stream(object):
        pass

    class ndarray(object):
        pass

# ------------------------------------------------------------------------------
# Global states
# ------------------------------------------------------------------------------
if available:
    cuda.set_default_allocator(cuda.MemoryPool().malloc)


def init(arg=None):
    """Obsolete function.

    Use :func:`~chainer.cuda.use_device` instead.

    """
    _check_cuda_available()
    use_device(arg)


def _check_cuda_available():
    if not available:
        global _resolution_error
        msg = 'CUDA environment is not correctly set up.\n'
        msg += str(_resolution_error)
        raise RuntimeError(msg)


def get_device(arg=None):
    """Gets the device from an ID integer or an array object.

    Args:
        arg: Value to specify a GPU device.

    Returns:
        Device object specified by given ``arg``.

        The rule of device selection is following.

==================================== =====================================
 Type of ``arg``                      Return value
==================================== =====================================
 ``None``                             Current device
 ``int``                              Device of ID ``arg``
 :class:`cupy.cuda.Device`            ``arg``
 :class:`cupy.ndarray`                Device given array was allocated on
 :class:`numpy.ndarray`               ``None``
==================================== =====================================

    """
    if arg is None:
        return Device()
    elif isinstance(arg, Device):
        return arg
    elif isinstance(arg, numpy.ndarray):
        return None
    elif isinstance(arg, cupy.ndarray):
        return arg.data.device
    else:
        return Device(arg)


def use_device(arg):
    """Switches the CUDA context to use given device.

    Args:
        arg: Argument of :func:`get_device`.

    """
    device = get_device(arg)
    if device is None:
        return
    device.use()


class DeviceUser(object):

    """RAII-style CUDA context swithcer.

    Args:
        arg: Argument of :func:`get_device`.

    Attributes:
        device (cupy.cuda.Device): Selected device.

    """
    def __init__(self, arg):
        if arg is None:
            self.device = None
        else:
            self.device = get_device(arg)
        if self.is_active:
            self.prev_device = Device()
        else:
            self.prev_device = None

    def __enter__(self):
        if self.is_active:
            self.device.use()
        return self

    def __exit__(self, typ, value, traceback):
        if self.prev_device is not None:
            self.prev_device.use()

    @property
    def is_active(self):
        return self.device is not None


def using_device(*args):
    """Returns a DeviceUser object of the first cupy.ndarray argument.

    If none of the arguments specifies a GPU device, then it returns a dummy
    :class:`DeviceUser` object which is inactive.

    Args:
        *args: Objects based on which an appropriate device should be selected.

    Returns:
        DeviceUser: Device user instance of selected argument.

    .. admonition:: Example

        Suppose ``arrays`` is a list of arrays of type either
        :class:`numpy.ndarray` or :class:`cupy.ndarray`. Then, the following
        code invokes ``do_something_on`` on an appropriate device::

            with using_device(*arrays):
                do_something_on(arrays)

    """
    for arg in args:
        dev = get_device(arg)
        if dev is not None:
            return DeviceUser(dev)
    return DeviceUser(None)


# ------------------------------------------------------------------------------
# cupy.ndarray allocation and copy
# ------------------------------------------------------------------------------

def to_gpu(array, device=None):
    """Copies the given CPU array to specified device.

    Args:
        array: Array to be sent to GPU.
        device: Device specifier.

    Returns:
        cupy.ndarray: Array on GPU.

        If ``array`` is already on GPU, then this function just returns
        ``array`` without performing any copy. Note that this function does not
        copy cupy.ndarray into specified device.

    """
    _check_cuda_available()
    if isinstance(array, cupy.ndarray):
        return array
    with using_device(device):
        return cupy.array(array)


def to_gpu_async(array, stream=None):
    """Copies the given CPU array asynchronously to the current device.

    Args:
        array: Array to be sent to GPU. If it is :class:`numpy.ndarray`, then
            its memory must be pagelocked.
        stream (cupy.cuda.Stream): CUDA stream.

    Returns:
        cupy.ndarray: Array on GPU.

        If given ``array`` is already on GPU, then this function just returns
        ``array`` without performing any copy.

    """
    _check_cuda_available()
    if isinstance(array, cupy.ndarray):
        return array
    assert stream is None
    return cupy.array(array)


def to_cpu(array):
    """Copies the given GPU array to host CPU.

    Args:
        array: Array to be sent to GPU.

    Returns:
        numpy.ndarray: Array on CPU.

        If given ``array`` is already on CPU, then this function just returns
        ``array`` without performing any copy.

    """
<<<<<<< HEAD
    _check_cuda_available()
    return cupy.asnumpy(array)
=======
    if isinstance(array, ndarray):
        return array.get()
    return array
>>>>>>> 54f157eda0730fcae809d8dcddf5dd8bbc0660bc


def to_cpu_async(array, stream=None):
    """Copies the given GPU array asynchronously to host CPU.

    Args:
        array: Array to be sent to GPU.
        stream (cupy.cuda.Stream): CUDA stream.

    Returns:
        numpy.ndarray: Array on CPU.

        If given ``array`` is already on CPU, then this function just returns
        ``array`` without performing any copy.

    """
<<<<<<< HEAD
    _check_cuda_available()
    return cupy.asnumpy(array, stream)
=======
    if isinstance(array, ndarray):
        return array.get(stream=stream)
    return array
>>>>>>> 54f157eda0730fcae809d8dcddf5dd8bbc0660bc


def empty(shape, dtype=numpy.float32):
    """Creates an uninitialized cupy.ndarray object.

    Args:
        shape (tuple of ints): The shape of array.
        dtype (numpy.dtype): Element type.

    Returns:
        cupy.ndarray: Uninitialized GPU array allocated by the memory pool.

    """
    _check_cuda_available()
    return cupy.empty(shape, dtype)


def full(shape, fill_value, dtype=numpy.float32, stream=None):
    """Creates a constant-filled cupy.ndarray object.

    Args:
        shape (tuple of ints): The shape of array.
        fill_value: Constant to fill the array by.
        dtype (numpy.dtype): Element type.
        stream (cupy.cuda.Stream): CUDA stream.

    Returns:
        cupy.ndarray: Constant-filled GPU array allocated by the memory pool.

    """
    _check_cuda_available()
    assert stream is None
    return cupy.full(shape, fill_value, dtype=dtype)


def zeros(shape, dtype=numpy.float32, stream=None):
    """Creates a zero-filled cupy.ndarray object.

    This function is equivalent to ``full(shape, 0, dtype, stream)``.

    """
    _check_cuda_available()
    assert stream is None
    return cupy.zeros(shape, dtype=dtype)


def ones(shape, dtype=numpy.float32, stream=None):
    """Creates a zero-filled cupy.ndarray object.

    This function is equivalent to ``full(shape, 1, dtype, stream)``.

    """
    _check_cuda_available()
    assert stream is None
    return cupy.ones(shape, dtype=dtype)


def empty_like(array):
    """Creates an uninitialized GPU array like the given one.

    Args:
        array (cupy.ndarray or numpy.ndarray): Base array.

    Returns:
        cupy.ndarray: GPU array of the same shape and dtype as `array`.

    """
    _check_cuda_available()
    if isinstance(array, cupy.ndarray):
        return cupy.empty_like(array)
    return cupy.empty(array.shape, dtype=array.dtype)


def full_like(array, fill_value, stream=None):
    """Creates a constant-filled cupy.ndarray object like the given array.

    Args:
        array (cupy.ndarray or numpy.ndarray): Base array.
        fill_value: Constant value to fill the array by.
        stream (cupy.cuda.Stream): CUDA stream.

    Returns:
        cupy.ndarray: Constant-filled array.

    """
    _check_cuda_available()
    assert stream is None
    if isinstance(array, cupy.ndarray):
        return cupy.full_like(array, fill_value)
    return cupy.full(array.shape, fill_value, dtype=array.dtype)


def zeros_like(array, stream=None):
    """Creates a zero-filled cupy.ndarray object like the given array.
<<<<<<< HEAD

    Args:
        array (cupy.ndarray or numpy.ndarray): Base array.
        stream (cupy.cuda.Stream): CUDA stream.

=======

    Args:
        array (cupy.ndarray or numpy.ndarray): Base array.
        stream (cupy.cuda.Stream): CUDA stream.

>>>>>>> 54f157eda0730fcae809d8dcddf5dd8bbc0660bc
    Returns:
        cupy.ndarray: Zero-filled array.

    """
    _check_cuda_available()
    assert stream is None
    if isinstance(array, cupy.ndarray):
        return cupy.zeros_like(array)
    return cupy.zeros(array.shape, dtype=array.dtype)


def ones_like(array, stream=None):
    """Creates a one-filled cupy.ndarray object like the given array.
<<<<<<< HEAD

    Args:
        array (cupy.ndarray or numpy.ndarray): Base array.
        stream (cupy.cuda.Stream): CUDA stream.

=======

    Args:
        array (cupy.ndarray or numpy.ndarray): Base array.
        stream (cupy.cuda.Stream): CUDA stream.

>>>>>>> 54f157eda0730fcae809d8dcddf5dd8bbc0660bc
    Returns:
        cupy.ndarray: One-filled array.

    """
    _check_cuda_available()
    assert stream is None
    if isinstance(array, cupy.ndarray):
        return cupy.ones_like(array)
    return cupy.ones(array.shape, dtype=array.dtype)


def copy(array, out=None, out_device=None):
    """Copies a cupy.ndarray object using the default stream.

    This function can copy the device array to the destination array on another
    device.

    Args:
        array (cupy.ndarray): Array to be copied.
        out (cupy.ndarray): Destination array.
            If it is not ``None``, then ``out_device`` argument is ignored.
        out_device: Destination device specifier. Actual device object is
            obtained by passing this value to :func:`get_device`.

    Returns:
        cupy.ndarray: Copied array.

        If ``out`` is not specified, then the array is allocated on the device
        specified by ``out_device`` argument.

    """
    _check_cuda_available()
    in_device = get_device(array)
    if out is None:
        if out_device is None:
            out_device = in_device
        else:
            out_device = get_device(out_device)

        with using_device(out_device):
            out = empty_like(array)
    else:
        out_device = get_device(out)

    with using_device(in_device):
        cupy.copyto(out, array)

    return out


def copy_async(array, out=None, out_device=None, stream=None):
    """Copies a cupy.ndarray object using the given stream.

    This function can copy the device array to the destination array on another
    device.

    Args:
        array (cupy.ndarray): Array to be copied.
        out (cupy.ndarray): Destination array.
            If it is not ``None``, then ``out_device`` argument is ignored.
        out_device: Destination device specifier. Actual device object is
            obtained by passing this value to :func:`get_device`.
        stream (cupy.cuda.Stream): CUDA stream.

    Returns:
        cupy.ndarray: Copied array.

        If ``out`` is not specified, then the array is allocated on the device
        specified by ``out_device`` argument.

    """
    _check_cuda_available()
    in_device = get_device(array)
    if out is None:
        if out_device is None:
            out_device = in_device
        else:
            out_device = get_device(out_device)

        with using_device(out_device):
            out = empty_like(array)
    else:
        out_device = get_device(out)

    with using_device(in_device):
        assert stream is None
        cupy.copyto(out, array)

    return out


# ------------------------------------------------------------------------------
# Kernel definition utility
# ------------------------------------------------------------------------------
def elementwise(param_names, operation, name, options=None,
                preamble='', loop_prep='', after_loop=''):
    """Creates an elementwise kernel function.

    This function uses :func:`cupy.cuda.memoize` to cache
    the resulting kernel object, i.e. the resulting kernel object is cached for
    each arguments and CUDA context.

    The arguments are the same as those for
    :func:`cupy.elementwise.ElementwiseKernel`, except that ``name`` argument
    is mandatory.

    """
    _check_cuda_available()
    return cupy.elementwise.ElementwiseKernel(
        param_names, operation, name, options,
        preamble=preamble, loop_prep=loop_prep, after_loop=after_loop)


def reduce(param_names, map_expr, reduce_expr, identity, name,
           dtype_out=numpy.float32, options=None,
           post_map_expr='a', preamble=''):
    """Creates a global reduction kernel function.

    This function uses :func:`cupy.cuda.memoize` to cache
    the resulting kernel object, i.e. the resulting kernel object is cached for
    each argument and CUDA context.

    The arguments are the same as those for
    :func:`cupy.reduction.ReductionKernel`, except that their order is
    different and ``name`` argument is mandatory.

    """
    _check_cuda_available()
    return cupy.reduction.ReductionKernel(
        dtype_out, param_names, identity, reduce_expr, map_expr, post_map_expr,
        name, options, preamble)


# ------------------------------------------------------------------------------
# numpy/cupy compatible coding
# ------------------------------------------------------------------------------
def get_xpy(a):
    """Gets an appropriate one from :mod:`numpy` or :mod:`cupy`.

    This function can be used to write a common ``forward`` and ``backward``
    running on both CPU and GPU.

    Args:
        a: An array of NumPy or CuPy.

    Returns:
        :mod:`numpy` module or :mod:`cupy` module corresponding to the type of
        ``a``.
    """
    if isinstance(a, numpy.ndarray):
        return numpy
    elif available and isinstance(a, ndarray):
        return cupy
    else:
        raise TypeError(
            'Cannot choose a NumPy-compatible module for {}'.format(type(a)))
