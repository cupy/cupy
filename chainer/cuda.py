"""Device, context and memory management on CuPy.

Chainer uses CuPy facilities (with very thin wrapper) to exploit the speed of
GPU computation. Following modules and classes are imported to :mod:`cuda`
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

Chainer provides thin wrappers of cupy.ndarray allocation routines, which use
:func:`mem_alloc` as the allocator. This allocator uses device-wise instance of
:class:`~pycuda.tools.DeviceMemoryPool`, which enables the reuse of device
memory over multiple forward/backward computations. :func:`mem_alloc` also
inserts an additional attribute to the allocated memory called ``device``,
which indicates the device that the memory is allocated on. Functions of
:mod:`cuda` uses this attribute to select appropriate device on each
manipulation routine.

"""
import numpy

_requires = []
try:
    import cupy
    import cupy.cuda
    from cupy import random
    from cupy.cuda import cublas
    available = True
except Exception as e:
    available = False
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
    pass


def _check_cuda_available():
    if not available:
        global _resolution_error
        msg = 'CUDA environment is not correctly set up.\n'
        msg += str(_resolution_error)
        raise RuntimeError(msg)


def get_device(arg=None):
    """Gets the device from ID arg or given chainer's.

    :class:`~cupy.ndarray`.

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
 :class:`~cupy.cuda.Device`           ``arg``
 :class:`~cupy.ndarray`               Device given array was allocated on
 :class:`~numpy.ndarray`              ``None``
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
        device (~pycuda.driver.Device): Selected device.
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
        :class:`~numpy.ndarray` or :class:`~cupy.ndarray`. Then,
        the following code invokes ``do_something_on`` with an appropriate
        context::

            with using_device(*arrays):
                do_something_on(arrays)

    """
    for arg in args:
        dev = get_device(arg)
        if dev is not None:
            return DeviceUser(dev)
    return DeviceUser(None)


def mem_alloc(nbytes):
    """Allocates device memory of given size from memory pool.

    This function chooses memory pool corresponding to the current device.

    Args:
        nbytes (int): The size of memory in bytes.

    Returns:
        cupy.cuda.MemoryPointer: Allocated memory

    """
    return cuda.alloc(nbytes)


#def _get_seed_getter(s=None):
#    if s is None:
#        return curandom.seed_getter_uniform
#    else:
#        return lambda N: full((N,), s, numpy.int32)
#
#
#def get_generator(device=None):
#    """Gets the random number generator for the given device.
#
#    Args:
#        device: Device specifier (an arugment of :func:`get_device`)
#
#    Returns:
#        pycuda.curandom.XORWOWRandomNumberGenerator: Random number generator.
#
#    """
#    global _generators
#
#    device = get_device(device)
#    gen = _generators.get(device)
#    if gen is not None:
#        return gen
#
#    with using_device(device):
#        s = os.environ.get('CHAINER_SEED')
#        seed_getter = _get_seed_getter(s)
#        gen = curandom.XORWOWRandomNumberGenerator(seed_getter=seed_getter)
#        _generators[device] = gen
#        return gen
#
#
#def seed(s=None, device=None):
#    """Resets the random number generator of the specified device.
#
#    Args:
#        s (int or None): Seed value. If it is ``None``, it initializes the
#            generator without fixed seed.
#        device: Device specifier (i.e. argument of :func:`get_device`).
#
#    """
#    global _generators
#
#    with DeviceUser(device) as user:
#        seed_getter = _get_seed_getter(s)
#        gen = curandom.XORWOWRandomNumberGenerator(seed_getter=seed_getter)
#        _generators[user.device] = gen


# ------------------------------------------------------------------------------
# cupy.ndarray allocation and copy
# ------------------------------------------------------------------------------

def to_gpu(array, device=None):
    """Copies the given CPU array to specified device.

    Args:
        array: Array to be sent to GPU.
        device: Device specifier.

    Returns:
        ~cupy.ndarray: Array on GPU.

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
        array: Array to be sent to GPU. If it is :class:`~numpy.ndarray`, then
            its memory must be pagelocked.
        stream (~cupy.cuda.Stream): CUDA stream.

    Returns:
        ~pycuda.cupy.ndarray: Array on GPU.

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
        ~numpy.ndarray: Array on CPU.

        If given ``array`` is already on CPU, then this function just returns
        ``array`` without performing any copy.

    """
    _check_cuda_available()
    return cupy.asnumpy(array)


def to_cpu_async(array, stream=None):
    """Copies the given GPU array asynchronously to host CPU.

    Args:
        array: Array to be sent to GPU.
        stream (~cupy.cuda.Stream): CUDA stream.

    Returns:
        ~numpy.ndarray: Array on CPU.

        If given ``array`` is already on CPU, then this function just returns
        ``array`` without performing any copy.

    """
    _check_cuda_available()
    return cupy.asnumpy(array, stream)


def empty(shape, dtype=numpy.float32):
    """Creates an uninitialized cupy.ndarray object.

    Args:
        shape (tuple of ints): The shape of array.
        dtype (numpy.dtype): Element type.

    Returns:
        ~cupy.ndarray: Uninitialized GPU array allocated by memory
        pool.

    """
    _check_cuda_available()
    return cupy.empty(shape, dtype)


def full(shape, fill_value, dtype=numpy.float32, stream=None):
    """Creates a constant-filled cupy.ndarray object.

    Args:
        shape (tuple of ints): The shape of array.
        fill_value: Constant to fill the array by.
        dtype (numpy.dtype): Element type.
        stream (~cupy.cuda.Stream): CUDA stream.

    Returns:
        ~pycuda.ndarray: Constant-filled GPU array allocated by
        memory pool.

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
    return zeros(shape, dtype=dtype)


def ones(shape, dtype=numpy.float32, stream=None):
    """Creates a zero-filled cupy.ndarray object.

    This function is equivalent to ``full(shape, 1, dtype, stream)``.

    """
    _check_cuda_available()
    assert stream is None
    return cupy.ones(shape, dtype=dtype)


def empty_like(array):
    """Alias to :func:`cupy.empty_like`."""
    _check_cuda_available()
    if isinstance(array, cupy.ndarray):
        return cupy.empty_like(array)
    return cupy.empty(array.shape, dtype=array.dtype)


def full_like(array, fill_value, stream=None):
    """Creates a constant-filled cupy.ndarray object like the given array.

    Args:
        array (~cupy.ndarray): Base array.
        fill_value: Constant value to fill the array by.
        stream (~cupy.cuda.Stream): CUDA stream.

    Returns:
        ~cupy.ndarray: Constant-filled array.

    """
    _check_cuda_available()
    assert stream is None
    if isinstance(array, cupy.ndarray):
        return cupy.full_like(array, fill_value)
    return cupy.full(array.shape, fill_value, dtype=array.dtype)


def zeros_like(array, stream=None):
    """Creates a zero-filled cupy.ndarray object like the given array.

    This function is equivalent to ``full_like(array, 0, stream)``.

    """
    _check_cuda_available()
    assert stream is None
    if isinstance(array, cupy.ndarray):
        return cupy.zeros_like(array)
    return cupy.zeros(array.shape, dtype=array.dtype)


def ones_like(array, stream=None):
    """Creates a one-filled cupy.ndarray object like the given array.

    This function is equivalent to ``full_like(array, 1, stream)``.

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
        array (~cupy.ndarray): Array to be copied.
        out (~cupy.ndarray): Destination array.
            If it is not ``None``, then ``out_device`` argument is ignored.
        out_device: Destination device specifier. Actual device object is
            obtained by passing this value to :func:`get_device`.

    Returns:
        ~cupy.ndarray: Copied array.

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
        array (~cupy.ndarray): Array to be copied.
        out (~cupy.ndarray): Destination array.
            If it is not ``None``, then ``out_device`` argument is ignored.
        out_device: Destination device specifier. Actual device object is
            obtained by passing this value to :func:`get_device`.
        stream (~cupy.cuda.Stream): CUDA stream.

    Returns:
        ~cupy.ndarray: Copied array.

        If ``out`` is not specified, then the array is allocated on the device
        specified by ``out_device`` argument.

    .. warning::

       Currently, copy_async over different devices raises exception, since
       PyCUDA drops the definition of :func:`pycuda.driver.memcopy_peer_async`.

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

## -----------------------------------------------------------------------------#-
## Interprocess communication
## -----------------------------------------------------------------------------#-
#
#
#class IPCEvent(Event):
#
#    """Event object for interprocess synchronization on GPU."""
#
#    def __init__(self):
#        super(IPCEvent, self).__init__(
#            drv.event_flags.INTERPROCESS | drv.event_flags.DISABLE_TIMING)
#
#
#class IPCArrayHandle(object):
#
#    """Converter between cupy.ndarray and its Inter-Process Communication handle.
#
#    It holds IPC memory handle with shape and dtype information. The instance
#    can be pickled, which means it can be passed through IPC path way, e.g.
#    Pipe and Queue. The other process can extract shared cupy.ndarray by calling
#    :meth:`get`. Also, the extracted array can be re-converted into another
#    IPCArrayHandle.
#
#    """
#
#    def __init__(self, array):
#        """Creates an IPC memory handle of the device array.
#
#        Args:
#            array (~cupy.ndarray): GPU array to be shared
#                accross processes.
#
#        """
#        if isinstance(array, drv.IPCMemoryHandle):
#            # do not doubly extract IPC memory handle
#            self.handle = array.ipc_handle
#        else:
#            self.handle = drv.mem_get_ipc_handle(array.ptr)
#
#        self.shape = array.shape
#        self.dtype = array.dtype
#        self.size = array.size
#        self.mem_size = array.mem_size
#
#    def get(self):
#        """Creates a cupy.ndarray object from the IPC memory handle.
#
#        Returns:
#            ~cupy.ndarray: Recovered GPU array with memory shared
#            accross processes.
#
#        .. note::
#
#           Note that :mod:`cuda` does not take care of data race between
#           multiple processes.
#
#        """
#        drv.IPCMemoryHandle(self.handle)
#        array = cupy.ndarray((0,), dtype=self.dtype)
#        array.shape = self.shape
#        array.size = self.size
#        array.mem_size = self.mem_size
#        setattr(array, 'ipc_handle', self.handle)
#        return array


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
           dtype_out=numpy.float32, keep=False, options=None, preamble=''):
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
        dtype_out, param_names, identity, reduce_expr, map_expr,
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
