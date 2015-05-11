"""Device, context and memory management on PyCUDA and scikits.cuda.

Chainer uses PyCUDA facilities (with very thin wrapper) to get speed of GPU
computation. Following modules and classes are imported to :mod:`cuda` module
for convenience (refer this table when reading chainer's source codes).

============================ =================================
 imported name                original name
============================ =================================
 ``chainer.cuda.cublas``      :mod:`scikits.cuda.cublas`
 ``chainer.cuda.curandom``    :mod:`pycuda.curandom`
 ``chainer.cuda.cumisc``      :mod:`scikits.cuda.misc`
 ``chainer.cuda.gpuarray``    :mod:`pycuda.gpuarray`

 ``chainer.cuda.Context``     :mod:`pycuda.driver.Context`
 ``chainer.cuda.Device``      :mod:`pycuda.driver.Device`
 ``chainer.cuda.Event``       :mod:`pycuda.driver.Event`
 ``chainer.cuda.GPUArray``    :mod:`pycuda.gpuarray.GPUArray`
 ``chainer.cuda.Stream``      :mod:`pycuda.driver.Stream`
============================ =================================

Chainer provides thin wrappers of GPUArray allocation routines, which use
:func:`mem_alloc` as the allocator. This allocator uses device-wise instance of
:class:`~pycuda.tools.DeviceMemoryPool`, which enables us to reuse device memory
over multiple forward/backward computations. :func:`mem_alloc` also inserts an
additional attribute to the allocated memory called ``device``, which indicates
the device that the memory is allocated on. Functions of :mod:`cuda` uses this
attribute to select appropriate device on each manipulation routine.

"""
import atexit, copy_reg, os
import numpy
import pycuda.driver as drv
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction   import ReductionKernel
from pycuda import gpuarray
from pycuda import curandom
import pycuda.tools as cutools
from scikits.cuda import cublas
import scikits.cuda.misc as cumisc

# ------------------------------------------------------------------------------
# Basic types
# ------------------------------------------------------------------------------
from pycuda.driver   import Context, Device, Event, Stream
from pycuda.gpuarray import GPUArray

# ------------------------------------------------------------------------------
# Global states
# ------------------------------------------------------------------------------
generator = None

_contexts = {}
_pools    = {}
_cublas_handles = {}
_pid      = None

def init(device=None):
    """Initializes CUDA global state.

    Chainer maintains CUDA context, CUBLAS context, random number generator and
    device memory pool for each GPU device and for each process (the main
    process or a process forked by :mod:`multiprocessing`). When this function
    is called first time on the process, it initializes these global states.

    .. warning::

       This function also initializes PyCUDA and scikits.cuda. Since these
       packages do not support forking after initialization, do not call this
       function before forking the process.

    This function also registers :func:`shutdown` to :mod:`atexit` slot.

    It also initializes random number generator. User can set fixed seed with
    ``CHAINER_SEED`` environment variable.

    Args:
        device (``int`` or :class:`~pycuda.driver.Device` or ``None``): Device
            ID to initialize on.

    """
    global generator, _contexts, _cublas_handles, _pid, _pools

    pid = os.getpid()
    if _pid == pid:  # already initialized
        return

    drv.init()

    if device is None:  # use default device
        context = cutools.make_default_context()
        device  = Context.get_device()
    else:
        device  = Device(device)
        context = device.make_context()
    _contexts  = {device: context}
    _pools = {}
    _cublas_handles = {}
    cumisc.init(mem_alloc)

    seed(os.environ.get('CHAINER_SEED'))

    _pid = pid  # mark as initialized
    atexit.register(shutdown)


def shutdown():
    """Finalize CUDA global state.

    This function is automatically called by :mod:`atexit`. Multiple calls is
    allowed, so user can manually call it if necessary.

    """
    global _contexts, _cublas_handles, _pid, _pools

    pid = os.getpid()
    if _pid != pid:  # not initialized
        return

    for cublas_handle in _cublas_handles.itervalues():
        cublas.cublasDestroy(cublas_handle)
    _cublas_handles = {}

    cumisc.shutdown()

    _pools = {}

    for ctx in _contexts.itervalues():
        ctx.detach()
    _contexts = {}
    _pid      = None  # mark as uninitialized


def get_device(arg=None):
    """Gets the device from ID or given chainer's
    :class:`~pycuda.gpuarray.GPUArray`.

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
         :class:`~pycuda.driver.Device`       ``arg``
         :class:`~pycuda.gpuarray.GPUArray`   Device given array was allocated on
         :class:`~numpy.ndarray`              ``None``
        ==================================== =====================================

    """
    if arg is None:
        return Context.get_device()
    elif isinstance(arg, drv.Device):
        return arg
    elif isinstance(arg, numpy.ndarray):
        return None
    elif isinstance(arg, GPUArray):
        return arg.gpudata.device
    return drv.Device(arg)


def use_device(arg, pop=True):
    """Swithces the CUDA context to use given device.

    Args:
        arg: Argument of :func:`get_device`.
        pop (bool): If True, this function pops the current context from context
            stack.

    """
    device = get_device(arg)
    if device is None:
        return

    if pop:
        drv.Context.pop()

    if device not in _contexts:
        _contexts[device] = device.make_context()
    else:
        _contexts[device].push()


class DeviceUser(object):
    """RAII-style CUDA context swithcer.

    Attributes:
        arg: Argument of :func:`use_device`.
    
    """
    def __init__(self, arg):
        self.arg = arg

    def __enter__(self):
        if self.is_active:
            use_device(self.arg, pop=False)
        return self

    def __exit__(self, typ, value, traceback):
        if self.is_active:
            drv.Context.pop()
        self.arg = None

    @property
    def is_active(self):
        return self.arg is not None

def using_device(*args):
    """Returns :class:`DeviceUser` object of the first
    :class:`~pycuda.gpuarray.GPUArray` argument.

    If none of the arguments is :class:`~pycuda.gpuarray.GPUArray`, then it
    returns dummy :class:`DeviceUser` object which is inactive.

    Args:
        *args: Objects based on which an appropriate device should be selected.

    Returns:
        DeviceUser: Device user instance of selected argument.

    .. admonition:: Example

        Suppose ``arrays`` is a list of arrays of type either
        :class:`~numpy.ndarray` or :class:`~pycuda.gpuarray.GPUArray`. Then,
        following code invokes ``do_something_on`` with an appropriate context::

            with using_device(*arrays):
                do_something_on(arrays)

    """
    for arg in args:
        if isinstance(arg, GPUArray):
            return DeviceUser(arg)
    return DeviceUser(None)


def get_context(arg=None):
    """Gets the context corresponding to the specified device.

    Args:
        arg: Argument of :func:`get_device`.

    Returns:
        ~pycuda.driver.Context: Context object corresponding to the specified
        device.

    """
    device = get_device(arg)
    if device is None:
        return None
    return _contexts[device]


def mem_alloc(nbytes):
    """Allocates device memory of given size from memory pool.

    This function chooses memory pool corresponding to the current device.

    Args:
        nbytes (int): The size of memory in bytes.

    Returns:
        pycuda.tools.PooledDeviceAllocation: Allocated memory with additional
        ``device`` attribute. This attribute is used to determine on which GPU
        the memory resides.

    """
    global _pools

    device = Context.get_device()
    pool   = _pools.get(device, None)

    if pool is None:
        pool = drv.DeviceMemoryPool()
        _pools[device] = pool

    allocation = pool.allocate(nbytes)
    setattr(allocation, 'device', device)
    return allocation


def seed(s=None):
    """Resets the random number generator by given seed.

    Args:
        s (``int`` or ``None``): Seed value. If it is ``None``, it initializes
            the generator without fixed seed.
    
    """
    global generator

    def seed_getter(N):
        if s is None:
            return curandom.seed_getter_uniform(N)
        return full((N,), s, numpy.int32)

    generator = curandom.XORWOWRandomNumberGenerator(seed_getter=seed_getter)


# ------------------------------------------------------------------------------
# GPUArray allocation and copy
# ------------------------------------------------------------------------------
# Workaround: the original GPUArray.copy does not use the user-defined
# allocator, so we have to replace it. A good solution is to inherit GPUArray
# and override copy method, but since many functions of pycuda.gpuarray directly
# use the original GPUArray class, we choose easy and ugly solution that
# directly replaces the original method.
# TODO(beam2d): Fix this ugly solution
def _gpuarray_copy(array):
    if not array.flags.forc:
        raise RuntimeError('only contiguous arrays may copied.')

    new = GPUArray(array.shape, array.dtype, allocator=array.allocator)
    drv.memcpy_dtod(new.gpudata, array.gpudata, array.nbytes)
    return new

GPUArray.copy = _gpuarray_copy


def to_gpu(array):
    """Copies given CPU array to the current device.

    Args:
        array (:class:`~numpy.ndarray` or :class:`~pycuda.gpuarray.GPUArray`):
            Array to be sent to GPU.

    Returns:
        ~pycuda.gpuarray.GPUArray: Array on GPU.

        If ``array`` is already on GPU, then this function just returns
        ``array`` without any copy.

    """
    if isinstance(array, GPUArray):
        return array
    return gpuarray.to_gpu(array, allocator=mem_alloc)

# Pickle redefinition of GPUArray. Note that pickling and unpickling of GPUArray
# do not preserve device information, i.e. the unpickled GPUArray may reside on
# a GPU different from the GPU that the original has resided on.
def _reconstruct(array, is_chainer_array):
    if is_chainer_array:
        return to_gpu(array)
    return gpuarray.to_gpu(array)

copy_reg.pickle(
    GPUArray,
    lambda data: (_reconstruct, (data.get(), hasattr(data.gpudata, 'device'))),
    _reconstruct)


def to_gpu_async(array, stream=None):
    """Copies given CPU array asynchronously to the current device.

    Args:
        array (:class:`~numpy.ndarray` or :class:`~pycuda.gpuarray.GPUArray`):
            Array to be sent to GPU. If it is :class:`~numpy.ndarray`, then its
            memory must be pagelocked.
        stream (:class:`~pycuda.driver.Stream` or ``None``): CUDA stream.

    Returns:
        ~pycuda.gpuarray.GPUArray: Array on GPU.

        If given ``array`` is already on GPU, then this function just returns
        ``array`` without any copy.

    """
    if isinstance(array, GPUArray):
        return array
    return gpuarray.to_gpu_async(array, allocator=mem_alloc, stream=stream)


def to_cpu(array):
    """Copies given GPU array to host CPU.

    Args:
        array (:class:`~numpy.ndarray` or :class:`~pycuda.gpuarray.GPUArray`):
            Array to be sent to GPU.

    Returns:
        ~numpy.ndarray: Array on CPU.

        If given ``array`` is already on CPU, then this function just returns
        ``array`` without any copy.

    """
    if isinstance(array, GPUArray):
        return array.get()
    return array


def to_cpu_async(array, stream=None):
    """Copies given GPU array asynchronously to host CPU.

    Args:
        array (:class:`~numpy.ndarray` or :class:`~pycuda.gpuarray.GPUArray`):
            Array to be sent to GPU.
        stream (:class:`~pycuda.driver.Stream` or ``None``): CUDA stream.

    Returns:
        ~numpy.ndarray: Array on CPU.

        If given ``array`` is already on CPU, then this function just returns
        ``array`` without any copy.

    """
    if isinstance(array, numpy.ndarray):
        return array
    return array.get_async(stream=stream)


def empty(shape, dtype=numpy.float32):
    """Creates an uninitialized :class:`~pycuda.gpuarray.GPUArray`.

    Args:
        shape (tuple of ints): The shape of array.
        dtype (:class:`numpy.dtype`): Element type.

    Returns:
        ~pycuda.gpuarray.GPUArray: Uninitialized GPU array allocated by memory
        pool.

    """
    return gpuarray.empty(shape, dtype, allocator=mem_alloc)


def full(shape, fill_value, dtype=numpy.float32, stream=None):
    """Creates a constan-filled :class:`~pycuda.gpuarray.GPUArray`.

    Args:
        shape (tuple of ints): The shape of array.
        fill_value: Constant to fill the array by.
        dtype (:class:`numpy.dtype`): Element type.
        stream (:class:`~pycuda.driver.Stream` or ``None``): CUDA stream.

    Returns:
        ~pycuda.gpuarray.GPUArray: Constant-filled GPU array allocated by memory
        pool.

    """
    array = empty(shape, dtype)
    array.fill(fill_value, stream=stream)
    return array


def zeros(shape, dtype=numpy.float32, stream=None):
    """Creates a zero-filled :class:`~pycuda.gpuarray.GPUArray`.

    This function is equivalent to ``full(shape, 0, dtype, stream)``.

    .. seealso:: full

    """
    return full(shape, 0, dtype, stream=stream)


def ones(shape, dtype=numpy.float32, stream=None):
    """Creates a zero-filled :class:`~pycuda.gpuarray.GPUArray`.

    This function is equivalent to ``full(shape, 1, dtype, stream)``.

    .. seealso:: full

    """
    return full(shape, 1, dtype, stream=stream)


empty_like = gpuarray.empty_like
"""Alias to :func:`pycuda.gpuarray.empty_like`."""


def full_like(array, fill_value, stream=None):
    """Creates a constant-filled :class:`~pycuda.gpuarray.GPUArray` like
    given array.

    Args:
        array (:class:`~pycuda.gpuarray.GPUArray`): Base array.
        fill_value: Constant value to fill the array by.
        stream (:class:`~pycuda.driver.Stream` or ``None``): CUDA stream.

    Returns:
        ~pycuda.gpuarray.GPUArray: Constant-filled array.

    """
    array = empty_like(array)
    array.fill(fill_value, stream=stream)
    return array


def zeros_like(array, stream=None):
    """Creates a zero-filled :class:`~pycuda.gpuarray.GPUArray` like
    given array.

    This function is equivalent to ``full_like(array, 0, stream)``.

    """
    return full_like(array, 0, stream=stream)


def ones_like(array, stream=None):
    """Creates a one-filled :class:`~pycuda.gpuarray.GPUArray` like
    given array.

    This function is equivalent to ``full_like(array, 1, stream)``.

    """
    return full_like(array, 1, stream=stream)
    

def copy(array, out=None, out_device=None):
    """Copies :class:`~pycuda.gpuarray.GPUArray` using default stream.

    This function can copy the device array to the destination array on another
    device.

    Args:
        array (:class:`~pycuda.gpuarray.GPUArray`): Array to be copied.
        out (:class:`~pycuda.gpuarray.GPUArray` or ``None``): Destination array.
            If it is not ``None``, then ``out_device`` argument is ignored.
        out_device: Destination device specifier. Actual device object is
            obtained by passing this value to :func:`get_device`.

    Returns:
        ~pycuda.gpuarray.GPUArray: Copied array.

        If ``out`` is not specified, then the array is allocated on the device
        specified by ``out_device`` argument.

    """
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
        if in_device == out_device:
            drv.memcpy_dtod(out.ptr, array.ptr, out.nbytes)
        else:
            drv.memcpy_peer(out.ptr, array.ptr, out.nbytes, out_device, in_device)

    return out


def copy_async(array, out=None, out_device=None, stream=None):
    """Copies :class:`~pycuda.gpuarray.GPUArray` using given stream.

    This function can copy the device array to the destination array on another
    device.

    Args:
        array (:class:`~pycuda.gpuarray.GPUArray`): Array to be copied.
        out (:class:`~pycuda.gpuarray.GPUArray` or ``None``): Destination array.
            If it is not ``None``, then ``out_device`` argument is ignored.
        out_device: Destination device specifier. Actual device object is
            obtained by passing this value to :func:`get_device`.
        stream (:class:`~pycuda.driver.Stream` or ``None``): CUDA stream.

    Returns:
        ~pycuda.gpuarray.GPUArray: Copied array.

        If ``out`` is not specified, then the array is allocated on the device
        specified by ``out_device`` argument.

    .. warning::

       Currently, copy_async over different devices raises exception, since
       PyCUDA drops the definition of :func:`pycuda.driver.memcopy_peer_async`.

    """
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
        if in_device == out_device:
            drv.memcpy_dtod_async(out.ptr, array.ptr, out.nbytes, stream=stream)
        else:
            drv.memcpy_peer_async(out.ptr, array.ptr, out.nbytes, out_device,
                                  in_device, stream=stream)

    return out


# ------------------------------------------------------------------------------
# Interprocess communication
# ------------------------------------------------------------------------------
class IPCEvent(Event):
    """:class:`~pycuda.driver.Event` object for interprocess synchronization on
    GPU."""

    def __init__(self):
        super(IPCEvent, self).__init__(
            drv.event_flags.INTERPROCESS | drv.event_flags.DISABLE_TIMING)


class IPCArrayHandle(object):
    """Converter between :class:`~pycuda.gpuarray.GPUArray` and its
    Inter-Process Communication handle.

    It holds IPC memory handle with shape and dtype information. The instance
    can be pickled, which means it can be passed through IPC path way, e.g. Pipe
    and Queue. The other process can extract shared GPUArray by calling
    :meth:`get`. Also, the extracted array can be re-converted into another
    IPCArrayHandle.

    """
    def __init__(self, array):
        """Creates an IPC memory handle of the device array.

        Args:
            array (:class:`~pycuda.gpuarray.GPUArray`): GPU array to be shared
                accross processes.

        """
        if isinstance(array, drv.IPCMemoryHandle):
            # do not doubly extract IPC memory handle
            self.handle = array.ipc_handle
        else:
            self.handle = drv.mem_get_ipc_handle(array.ptr)

        self.shape    = array.shape
        self.dtype    = array.dtype
        self.size     = array.size
        self.mem_size = array.mem_size

    def get(self):
        """Creates :class:`~pycuda.gpuarray.GPUArray` from IPC memory handle.

        Returns:
            ~pycuda.gpuarray.GPUArray: Recovered GPU array with memory shared
            accross processes.

        .. note::

           Note that :mod:`cuda` does not take care of data race between
           multiple processes.

        """
        mem = drv.IPCMemoryHandle(self.handle)
        array = gpuarray.GPUArray((0,), dtype=self.dtype)
        array.shape    = self.shape
        array.size     = self.size
        array.mem_size = self.mem_size
        setattr(array, 'ipc_handle', self.handle)
        return array


# ------------------------------------------------------------------------------
# Kernel definition utility
# ------------------------------------------------------------------------------
@cutools.context_dependent_memoize
def _eltwise_kernel(arguments, operation, name, keep, options,
                    preamble, loop_prep, after_loop):
    return ElementwiseKernel(
        arguments, operation, name, keep, options,
        preamble=preamble, loop_prep=loop_prep, after_loop=after_loop)

def elementwise(arguments, operation, name, keep=False, options=None,
                preamble='', loop_prep='', after_loop=''):
    """Creates an elementwise kernel function.

    This function uses :func:`pycuda.tools.context_dependent_memoize` to cache
    the resulting kernel object, i.e. the resulting kernel object is cached for
    each arguments and CUDA context.

    The arguments are same as :func:`pycuda.elementwise.ElementwiseKernel`,
    except that ``name`` argument is mandatory.

    .. seealso:: :class:`pycuda.elementwise.ElementwiseKernel`

    """
    return _eltwise_kernel(arguments, operation, name, keep, options,
                           preamble, loop_prep, after_loop)


@cutools.context_dependent_memoize
def _reduce_kernel(dtype_out, neutral, reduce_expr, map_expr, arguments,
                   name, keep, options, preamble):
    return ReductionKernel(dtype_out, neutral, reduce_expr, map_expr, arguments,
                           name, keep, options, preamble)

def reduce(arguments, map_expr, reduce_expr, neutral, name,
           dtype_out=numpy.float32, keep=False, options=None, preamble=''):
    """Creates a global reduction kernel function.

    This function uses :func:`pycuda.tools.context_dependent_memoize` to cache
    the resulting kernel object, i.e. the resulting kernel object is cached for
    each arguments and CUDA context.

    The arguments are same as :func:`pycuda.reduction.ReductionKernel`,
    except that the order is different and ``name`` argument is mandatory.

    .. seealso:: :class:`pycuda.reduction.ReductionKernel`

    """
    kern = _reduce_kernel(dtype_out, neutral, reduce_expr, map_expr, arguments,
                          name, keep, options, preamble)
    def call_kern(*args, **kwargs):
        kwargs['allocator'] = mem_alloc
        return kern(*args, **kwargs)
    return call_kern


# ------------------------------------------------------------------------------
# CUBLAS
# ------------------------------------------------------------------------------

def get_cublas_handle():
    """Gets CUBLAS handle for the current device.

    Returns:
        CUBLAS handle.

    """
    global _cublas_handles

    device = Context.get_device()
    if device in _cublas_handles:
        return _cublas_handles[device]

    handle = cublas.cublasCreate()
    _cublas_handles[device] = handle
    return handle


class CumiscUser(object):
    """RAII-style switcher of :mod:`scikits.cuda.misc` default CUBLAS handle.
    """

    def __init__(self, handle):
        """Initializes the misc user by given handle.

        Args:
            handle: CUBLAS handle.

        """
        self.handle = cumisc._global_cublas_handle
        self.tmp_handle = handle

    def __enter__(self):
        cumisc._global_cublas_handle = self.tmp_handle

    def __exit__(self, typ, value, traceback):
        cumisc._global_cublas_handle = self.handle


def using_cumisc(handle=None):
    """Temporarily use chainer's CUBLAS handle on :mod:`scikits.cuda.misc`.

    The usage is similar to :func:`using_device`.

    Args:
        handle: CUBLAS handle. If ``None`` is specified, it uses CUBLAS handle
            for the current device.

    Returns:
        CumiscUser: Misc user object.

    """
    if handle is None:
        handle = get_cublas_handle()
    return CumiscUser(handle)
