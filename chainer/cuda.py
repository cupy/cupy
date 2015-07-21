"""Device, context and memory management on PyCUDA and scikit-cuda.

Chainer uses PyCUDA facilities (with very thin wrapper) to exploit the speed of
GPU computation. Following modules and classes are imported to :mod:`cuda`
module for convenience (refer to this table when reading chainer's source
codes).

============================ =================================
 imported name                original name
============================ =================================
 ``chainer.cuda.cublas``      :mod:`skcuda.cublas`
 ``chainer.cuda.cumath``      :mod:`pycuda.cumath`
 ``chainer.cuda.curandom``    :mod:`pycuda.curandom`
 ``chainer.cuda.culinalg``    :mod:`skcuda.linalg`
 ``chainer.cuda.cumisc``      :mod:`skcuda.misc`
 ``chainer.cuda.gpuarray``    :mod:`pycuda.gpuarray`

 ``chainer.cuda.Context``     :mod:`pycuda.driver.Context`
 ``chainer.cuda.Device``      :mod:`pycuda.driver.Device`
 ``chainer.cuda.Event``       :mod:`pycuda.driver.Event`
 ``chainer.cuda.GPUArray``    :mod:`pycuda.gpuarray.GPUArray`
 ``chainer.cuda.Stream``      :mod:`pycuda.driver.Stream`
============================ =================================

Chainer provides thin wrappers of GPUArray allocation routines, which use
:func:`mem_alloc` as the allocator. This allocator uses device-wise instance of
:class:`~pycuda.tools.DeviceMemoryPool`, which enables the reuse of device
memory over multiple forward/backward computations. :func:`mem_alloc` also
inserts an additional attribute to the allocated memory called ``device``,
which indicates the device that the memory is allocated on. Functions of
:mod:`cuda` uses this attribute to select appropriate device on each
manipulation routine.

"""
import atexit
import os
import warnings

import numpy
import pkg_resources
import six

try:
    try:
        pkg_resources.require('scikits.cuda')
    except pkg_resources.ResolutionError as e:
        pass
    else:
        msg = '''
`scikits.cuda` package is found. This is deprecated.
Clean both the old and new `scikit-cuda` packages, and then re-install
`chainer-cuda-deps`.

$ pip uninstall scikits.cuda scikit-cuda
$ pip install -U chainer-cuda-deps
        '''
        warnings.warn(msg)

    _requires = [
        'pycuda>=2014.1',
        'scikit-cuda>=0.5.0',
        'Mako',
        'six>=1.9.0',
    ]
    pkg_resources.require(_requires)

    import pycuda.cumath
    import pycuda.curandom
    import pycuda.driver as drv
    import pycuda.elementwise
    import pycuda.gpuarray
    import pycuda.reduction
    import pycuda.tools
    import skcuda.cublas
    import skcuda.linalg
    import skcuda.misc
    available = True

    cublas = skcuda.cublas
    cumath = pycuda.cumath
    curandom = pycuda.curandom
    culinalg = skcuda.linalg
    cumisc = skcuda.misc
    cutools = pycuda.tools
    gpuarray = pycuda.gpuarray
except pkg_resources.ResolutionError as e:
    available = False
    _resolution_error = e

# ------------------------------------------------------------------------------
# Basic types
# ------------------------------------------------------------------------------
if available:
    import pycuda.driver
    import pycuda.gpuarray

    Context = pycuda.driver.Context
    Device = pycuda.driver.Device
    Event = pycuda.driver.Event
    Stream = pycuda.driver.Stream
    GPUArray = pycuda.gpuarray.GPUArray
else:
    # Dummy classes
    class Context(object):
        pass

    class Device(object):
        pass

    class Event(object):
        pass

    class Stream(object):
        pass

    class GPUArray(object):
        pass

# ------------------------------------------------------------------------------
# Global states
# ------------------------------------------------------------------------------
generator = None

_contexts = {}
_pools = {}
_generators = {}
_cublas_handles = {}
_pid = None


def _check_cuda_available():
    if not available:
        global _resolution_error
        msg = '''CUDA environment is not correctly set up.
Use `pip install -U chainer-cuda-deps` to install libraries.
'''

        # Note that error message depends on its type
        if isinstance(_resolution_error, pkg_resources.DistributionNotFound):
            msg += 'Required package is not found: ' + str(_resolution_error)
        elif isinstance(_resolution_error, pkg_resources.VersionConflict):
            msg += 'Version conflict: ' + str(_resolution_error)
        else:
            msg += 'Unknwon error: ' + str(_resolution_error)

        raise RuntimeError(msg)


def init(device=None):
    """Initializes CUDA global state.

    Chainer maintains CUDA context, CUBLAS context, random number generator and
    device memory pool for each GPU device and for each process (the main
    process or a process forked by :mod:`multiprocessing`) as global states.
    When called for the first time on the process, this function initializes
    these global states.

    .. warning::

       This function also initializes PyCUDA and scikit-cuda. Since these
       packages do not support forking after initialization, do not call this
       function before forking the process.

    This function also registers :func:`shutdown` to :mod:`atexit` slot.

    It also initializes random number generator. User can set fixed seed with
    ``CHAINER_SEED`` environment variable.

    Args:
        device (``int`` or :class:`~pycuda.driver.Device` or ``None``): Device
            ID to initialize on.

    """
    global _contexts, _cublas_handles, _generators, _pid, _pools

    _check_cuda_available()

    pid = os.getpid()
    if _pid == pid:  # already initialized
        return

    drv.init()

    if device is None:  # use default device
        context = cutools.make_default_context()
        device = Context.get_device()
    else:
        device = Device(device)
        context = device.make_context()
    _contexts = {device: context}
    _generators = {}
    _pools = {}
    _cublas_handles = {}
    cumisc.init(mem_alloc)

    seed(os.environ.get('CHAINER_SEED'))

    _pid = pid  # mark as initialized
    atexit.register(shutdown)


def shutdown():
    """Finalizes CUDA global state.

    This function is automatically called by :mod:`atexit`. Multiple calls are
    allowed, so user can manually call this function if necessary.

    """
    global _contexts, _cublas_handles, _pid, _pools

    _check_cuda_available()

    pid = os.getpid()
    if _pid != pid:  # not initialized
        return

    for cublas_handle in six.itervalues(_cublas_handles):
        cublas.cublasDestroy(cublas_handle)
    _cublas_handles = {}

    cumisc.shutdown()

    _pools = {}

    for ctx in six.itervalues(_contexts):
        ctx.detach()
    _contexts = {}
    _pid = None  # mark as uninitialized


def get_device(arg=None):
    """Gets the device from ID arg or given chainer's.

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
    elif isinstance(arg, Device):
        return arg
    elif isinstance(arg, numpy.ndarray):
        return None
    elif isinstance(arg, GPUArray):
        while not hasattr(arg.gpudata, 'device'):
            arg = arg.base
        return arg.gpudata.device
    return drv.Device(arg)


def use_device(arg, pop=True):
    """Switches the CUDA context to use given device.

    Args:
        arg: Argument of :func:`get_device`.
        pop (bool): If True, pop the current context from context
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

    def __enter__(self):
        if self.is_active:
            use_device(self.device, pop=False)
        return self

    def __exit__(self, typ, value, traceback):
        if self.is_active:
            drv.Context.pop()
        self.device = None

    @property
    def is_active(self):
        return self.device is not None


def using_device(*args):
    """Returns a DeviceUser object of the first GPUArray argument.

    If none of the arguments specifies a GPU device, then it returns a dummy
    :class:`DeviceUser` object which is inactive.

    Args:
        *args: Objects based on which an appropriate device should be selected.

    Returns:
        DeviceUser: Device user instance of selected argument.

    .. admonition:: Example

        Suppose ``arrays`` is a list of arrays of type either
        :class:`~numpy.ndarray` or :class:`~pycuda.gpuarray.GPUArray`. Then,
        the following code invokes ``do_something_on`` with an appropriate
        context::

            with using_device(*arrays):
                do_something_on(arrays)

    """
    for arg in args:
        user = DeviceUser(arg)
        if user.is_active:
            return user
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
    pool = _pools.get(device, None)

    if pool is None:
        pool = drv.DeviceMemoryPool()
        _pools[device] = pool

    allocation = pool.allocate(nbytes)
    setattr(allocation, 'device', device)
    return allocation


def _get_seed_getter(s=None):
    if s is None:
        return curandom.seed_getter_uniform
    else:
        return lambda N: full((N,), s, numpy.int32)


def get_generator(device=None):
    """Gets the random number generator for the given device.

    Args:
        device: Device specifier (an arugment of :func:`get_device`)

    Returns:
        pycuda.curandom.XORWOWRandomNumberGenerator: Random number generator.

    """
    global _generators

    device = get_device(device)
    gen = _generators.get(device)
    if gen is not None:
        return gen

    with using_device(device):
        s = os.environ.get('CHAINER_SEED')
        seed_getter = _get_seed_getter(s)
        gen = curandom.XORWOWRandomNumberGenerator(seed_getter=seed_getter)
        _generators[device] = gen
        return gen


def seed(s=None, device=None):
    """Resets the random number generator of the specified device.

    Args:
        s (int or None): Seed value. If it is ``None``, it initializes the
            generator without fixed seed.
        device: Device specifier (i.e. argument of :func:`get_device`).

    """
    global _generators

    with DeviceUser(device) as user:
        seed_getter = _get_seed_getter(s)
        gen = curandom.XORWOWRandomNumberGenerator(seed_getter=seed_getter)
        _generators[user.device] = gen


# ------------------------------------------------------------------------------
# GPUArray allocation and copy
# ------------------------------------------------------------------------------
# Workaround: the original GPUArray.copy does not use the user-defined
# allocator, so we have to replace it. A good solution is to inherit GPUArray
# and override copy method, but since many functions of pycuda.gpuarray
# directly use the original GPUArray class, we choose easy and ugly solution
# that directly replaces the original method.
# TODO(beam2d): Fix this ugly solution
def _gpuarray_copy(array):
    if not array.flags.forc:
        raise RuntimeError('only contiguous arrays may copied.')

    new = GPUArray(array.shape, array.dtype, allocator=array.allocator)
    drv.memcpy_dtod(new.gpudata, array.gpudata, array.nbytes)
    return new

GPUArray.copy = _gpuarray_copy


def to_gpu(array, device=None):
    """Copies the given CPU array to specified device.

    Args:
        array: Array to be sent to GPU.
        device: Device specifier.

    Returns:
        ~pycuda.gpuarray.GPUArray: Array on GPU.

        If ``array`` is already on GPU, then this function just returns
        ``array`` without performing any copy. Note that this function does not
        copy GPUArray into specified device.

    """
    _check_cuda_available()
    if isinstance(array, GPUArray):
        return array
    with using_device(device):
        return gpuarray.to_gpu(array, allocator=mem_alloc)

# Pickle redefinition of GPUArray. Note that pickling and unpickling of
# GPUArray do not preserve device information, i.e. the unpickled GPUArray may
# reside on a GPU different from the GPU that the original has resided on.


def _reconstruct(array, is_chainer_array):
    if is_chainer_array:
        return to_gpu(array)
    return gpuarray.to_gpu(array)

six.moves.copyreg.pickle(
    GPUArray,
    lambda data: (_reconstruct, (data.get(), hasattr(data.gpudata, 'device'))),
    _reconstruct)


def to_gpu_async(array, stream=None):
    """Copies the given CPU array asynchronously to the current device.

    Args:
        array: Array to be sent to GPU. If it is :class:`~numpy.ndarray`, then
            its memory must be pagelocked.
        stream (~pycuda.driver.Stream): CUDA stream.

    Returns:
        ~pycuda.gpuarray.GPUArray: Array on GPU.

        If given ``array`` is already on GPU, then this function just returns
        ``array`` without performing any copy.

    """
    _check_cuda_available()
    if isinstance(array, GPUArray):
        return array
    return gpuarray.to_gpu_async(array, allocator=mem_alloc, stream=stream)


def to_cpu(array):
    """Copies the given GPU array to host CPU.

    Args:
        array: Array to be sent to GPU.

    Returns:
        ~numpy.ndarray: Array on CPU.

        If given ``array`` is already on CPU, then this function just returns
        ``array`` without performing any copy.

    """
    if isinstance(array, GPUArray):
        return array.get()
    return array


def to_cpu_async(array, stream=None):
    """Copies the given GPU array asynchronously to host CPU.

    Args:
        array: Array to be sent to GPU.
        stream (~pycuda.driver.Stream): CUDA stream.

    Returns:
        ~numpy.ndarray: Array on CPU.

        If given ``array`` is already on CPU, then this function just returns
        ``array`` without performing any copy.

    """
    if isinstance(array, numpy.ndarray):
        return array
    return array.get_async(stream=stream)


def empty(shape, dtype=numpy.float32):
    """Creates an uninitialized GPUArray object.

    Args:
        shape (tuple of ints): The shape of array.
        dtype (numpy.dtype): Element type.

    Returns:
        ~pycuda.gpuarray.GPUArray: Uninitialized GPU array allocated by memory
        pool.

    """
    _check_cuda_available()
    return gpuarray.empty(shape, dtype, allocator=mem_alloc)


def full(shape, fill_value, dtype=numpy.float32, stream=None):
    """Creates a constant-filled GPUArray object.

    Args:
        shape (tuple of ints): The shape of array.
        fill_value: Constant to fill the array by.
        dtype (numpy.dtype): Element type.
        stream (~pycuda.driver.Stream): CUDA stream.

    Returns:
        ~pycuda.gpuarray.GPUArray: Constant-filled GPU array allocated by
        memory pool.

    """
    array = empty(shape, dtype)
    array.fill(fill_value, stream=stream)
    return array


def zeros(shape, dtype=numpy.float32, stream=None):
    """Creates a zero-filled GPUArray object.

    This function is equivalent to ``full(shape, 0, dtype, stream)``.

    """
    return full(shape, 0, dtype, stream=stream)


def ones(shape, dtype=numpy.float32, stream=None):
    """Creates a zero-filled GPUArray object.

    This function is equivalent to ``full(shape, 1, dtype, stream)``.

    """
    return full(shape, 1, dtype, stream=stream)


def empty_like(array):
    """Alias to :func:`pycuda.gpuarray.empty_like`."""
    _check_cuda_available()
    return gpuarray.empty_like(array)


def full_like(array, fill_value, stream=None):
    """Creates a constant-filled GPUArray object like the given array.

    Args:
        array (~pycuda.gpuarray.GPUArray): Base array.
        fill_value: Constant value to fill the array by.
        stream (~pycuda.driver.Stream): CUDA stream.

    Returns:
        ~pycuda.gpuarray.GPUArray: Constant-filled array.

    """
    array = empty_like(array)
    array.fill(fill_value, stream=stream)
    return array


def zeros_like(array, stream=None):
    """Creates a zero-filled GPUArray object like the given array.

    This function is equivalent to ``full_like(array, 0, stream)``.

    """
    return full_like(array, 0, stream=stream)


def ones_like(array, stream=None):
    """Creates a one-filled GPUArray object like the given array.

    This function is equivalent to ``full_like(array, 1, stream)``.

    """
    return full_like(array, 1, stream=stream)


def copy(array, out=None, out_device=None):
    """Copies a GPUArray object using the default stream.

    This function can copy the device array to the destination array on another
    device.

    Args:
        array (~pycuda.gpuarray.GPUArray): Array to be copied.
        out (~pycuda.gpuarray.GPUArray): Destination array.
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
            drv.memcpy_peer(
                out.ptr, array.ptr, out.nbytes, out_device, in_device)

    return out


def copy_async(array, out=None, out_device=None, stream=None):
    """Copies a GPUArray object using the given stream.

    This function can copy the device array to the destination array on another
    device.

    Args:
        array (~pycuda.gpuarray.GPUArray): Array to be copied.
        out (~pycuda.gpuarray.GPUArray): Destination array.
            If it is not ``None``, then ``out_device`` argument is ignored.
        out_device: Destination device specifier. Actual device object is
            obtained by passing this value to :func:`get_device`.
        stream (~pycuda.driver.Stream): CUDA stream.

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
            drv.memcpy_dtod_async(
                out.ptr, array.ptr, out.nbytes, stream=stream)
        else:
            drv.memcpy_peer_async(out.ptr, array.ptr, out.nbytes, out_device,
                                  in_device, stream=stream)

    return out

# ------------------------------------------------------------------------------
# Add comparison of `__array_priority__` to GPUArray binary operator
# ------------------------------------------------------------------------------


def _wrap_operation(obj, op):
    op_name = '__{}__'.format(op)
    rop_name = '__r{}__'.format(op)
    raw_op = getattr(obj, op_name)

    def new_op(self, other):
        rop = getattr(other, rop_name, None)
        if rop is None:
            return raw_op(self, other)
        self_pri = getattr(self,  '__array_priority__', 0.0)
        other_pri = getattr(other, '__array_priority__', 0.0)
        if self_pri >= other_pri:
            return raw_op(self, other)
        return rop(self)
    setattr(obj, op_name, new_op)

if available:
    for op in ('add', 'sub', 'mul', 'div', 'pow', 'truediv'):
        _wrap_operation(GPUArray, op)

# ------------------------------------------------------------------------------
# Interprocess communication
# ------------------------------------------------------------------------------


class IPCEvent(Event):

    """Event object for interprocess synchronization on GPU."""

    def __init__(self):
        super(IPCEvent, self).__init__(
            drv.event_flags.INTERPROCESS | drv.event_flags.DISABLE_TIMING)


class IPCArrayHandle(object):

    """Converter between GPUArray and its Inter-Process Communication handle.

    It holds IPC memory handle with shape and dtype information. The instance
    can be pickled, which means it can be passed through IPC path way, e.g.
    Pipe and Queue. The other process can extract shared GPUArray by calling
    :meth:`get`. Also, the extracted array can be re-converted into another
    IPCArrayHandle.

    """

    def __init__(self, array):
        """Creates an IPC memory handle of the device array.

        Args:
            array (~pycuda.gpuarray.GPUArray): GPU array to be shared
                accross processes.

        """
        if isinstance(array, drv.IPCMemoryHandle):
            # do not doubly extract IPC memory handle
            self.handle = array.ipc_handle
        else:
            self.handle = drv.mem_get_ipc_handle(array.ptr)

        self.shape = array.shape
        self.dtype = array.dtype
        self.size = array.size
        self.mem_size = array.mem_size

    def get(self):
        """Creates a GPUArray object from the IPC memory handle.

        Returns:
            ~pycuda.gpuarray.GPUArray: Recovered GPU array with memory shared
            accross processes.

        .. note::

           Note that :mod:`cuda` does not take care of data race between
           multiple processes.

        """
        drv.IPCMemoryHandle(self.handle)
        array = gpuarray.GPUArray((0,), dtype=self.dtype)
        array.shape = self.shape
        array.size = self.size
        array.mem_size = self.mem_size
        setattr(array, 'ipc_handle', self.handle)
        return array


# ------------------------------------------------------------------------------
# Kernel definition utility
# ------------------------------------------------------------------------------
if available:
    @cutools.context_dependent_memoize
    def _eltwise_kernel(arguments, operation, name, keep, options,
                        preamble, loop_prep, after_loop):
        return pycuda.elementwise.ElementwiseKernel(
            arguments, operation, name, keep, options,
            preamble=preamble, loop_prep=loop_prep, after_loop=after_loop)


def elementwise(arguments, operation, name, keep=False, options=None,
                preamble='', loop_prep='', after_loop=''):
    """Creates an elementwise kernel function.

    This function uses :func:`pycuda.tools.context_dependent_memoize` to cache
    the resulting kernel object, i.e. the resulting kernel object is cached for
    each arguments and CUDA context.

    The arguments are the same as those for
    :func:`pycuda.elementwise.ElementwiseKernel`, except that ``name`` argument
    is mandatory.

    """
    return _eltwise_kernel(arguments, operation, name, keep, options,
                           preamble, loop_prep, after_loop)


if available:
    @cutools.context_dependent_memoize
    def _reduce_kernel(dtype_out, neutral, reduce_expr, map_expr, arguments,
                       name, keep, options, preamble):
        return pycuda.reduction.ReductionKernel(
            dtype_out, neutral, reduce_expr, map_expr,
            arguments, name, keep, options, preamble)


def reduce(arguments, map_expr, reduce_expr, neutral, name,
           dtype_out=numpy.float32, keep=False, options=None, preamble=''):
    """Creates a global reduction kernel function.

    This function uses :func:`pycuda.tools.context_dependent_memoize` to cache
    the resulting kernel object, i.e. the resulting kernel object is cached for
    each argument and CUDA context.

    The arguments are the same as those for
    :func:`pycuda.reduction.ReductionKernel`, except that their order is
    different and ``name`` argument is mandatory.

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

    """RAII-style switcher of scikits-cuda's default CUBLAS handle."""

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
    """Temporarily set chainer's CUBLAS handle to scikit-cuda.

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
