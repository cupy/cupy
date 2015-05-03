"""Device, context and memory management on PyCUDA."""

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

class IPCEvent(Event):
    def __init__(self):
        super(IPCEvent, self).__init__(
            drv.event_flags.INTERPROCESS | drv.event_flags.DISABLE_TIMING)

# ------------------------------------------------------------------------------
# Global states
# ------------------------------------------------------------------------------
mem_alloc = None

_contexts = {}
_pools    = {}
_cublas_handles = {}
_pid      = None

_seed      = os.environ.get('CHAINER_SEED')
generator = None

def _seed_getter(N):
    if _seed is None:
        return curandom.seed_getter_uniform(N)
    return full((N,), _seed, numpy.int32)

def init(device=None):
    """Initialize CUDA global state.

    If you use chainer.cuda with multiprocessing, call init() for each process
    **after the fork** and before using chainer with PyCUDA.

    """
    global generator, _contexts, _pid

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

    generator = curandom.XORWOWRandomNumberGenerator(seed_getter=_seed_getter)

    _pid      = pid  # mark as initialized
    atexit.register(shutdown)

    cumisc.init(mem_alloc)
    _cublas_handles = {device: cumisc._global_cublas_handle}


def shutdown():
    """Finalize CUDA global state."""

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
    """Get device from id or chainer array."""

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
    """Switch context to use given device."""

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
    """Device user for 'with' statement."""

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
    """Return DeviceUser object of the first GPUArray argument.

    If none of the arguments is GPUArray, then it returns dummy DeviceUser that
    does nothing.

    """
    for arg in args:
        if isinstance(arg, GPUArray):
            return DeviceUser(arg)
    return DeviceUser(None)


def get_context(arg=None):
    """Get the context of given device.

    If arg is not specified, it returns the current context.

    """
    device = get_device(arg)
    if device is None:
        return None
    return _contexts[device]


def mem_alloc(nbytes):
    """Allocate memory from memory pool corresponding to the current device."""
    global _pools

    device = Context.get_device()
    pool   = _pools.get(device, None)

    if pool is None:
        pool = drv.DeviceMemoryPool()
        _pools[device] = pool

    allocation = pool.allocate(nbytes)
    setattr(allocation, 'device', device)
    return allocation


# ------------------------------------------------------------------------------
# GPUArray allocation and copy
# ------------------------------------------------------------------------------
def to_gpu(array):
    """Copy array to the current device."""
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
    """Copy array asynchronously to the current device."""
    if isinstance(array, GPUArray):
        return array
    return gpuarray.to_gpu_async(array, allocator=mem_alloc, stream=stream)


def to_cpu(array):
    """Copy array to host synchronously."""
    if isinstance(array, numpy.ndarray):
        return array
    return array.get()


def to_cpu_async(array, stream=None):
    """Copy array to host asynchronously."""
    if isinstance(array, numpy.ndarray):
        return array
    return arra.get_async(stream=stream)


def empty(shape, dtype=numpy.float32):
    """Create an uninitialized GPUArray."""
    return gpuarray.empty(shape, dtype, allocator=mem_alloc)


def full(shape, fill_value, dtype=numpy.float32, stream=None):
    """Create constant-filled GPUArray."""
    array = empty(shape, dtype)
    array.fill(fill_value, stream=stream)
    return array


def zeros(shape, dtype=numpy.float32, stream=None):
    """Create zero-filled GPUArray."""
    return full(shape, 0, dtype, stream=stream)


def ones(shape, dtype=numpy.float32, stream=None):
    """Create one-filled GPUArray."""
    return full(shape, 1, dtype, stream=stream)


empty_like = gpuarray.empty_like


def full_like(array, fill_value, stream=None):
    """Create a constant-filled GPUArray like ``array``."""
    array = empty_like(array)
    array.fill(fill_value, stream=stream)
    return array


def zeros_like(array, stream=None):
    """Create a zero-filled GPUArray like ``array``."""
    return full_like(array, 0, stream=stream)


def ones_like(array, stream=None):
    """Create a zero-filled GPUArray like ``array``."""
    return full_like(array, 1, stream=stream)
    

def copy(array, out=None, out_device=None):
    """Copy GPUArray in default stream."""

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
    """Copy GPUArray asynchronously."""

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


class IPCArrayHandle(object):
    """Converter between GPUArray and its Inter-Process Communication handle.

    It holds IPC memory handle with shape and dtype information. The instance
    can be pickled, which means it can be passed through IPC path way, e.g. Pipe
    and Queue. The other process can extract shared GPUArray by calling get().
    Also, the extracted array can be re-converted into IPCArrayHandle.

    """
    def __init__(self, array):
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
    """Return memoized elementwise kernel function."""
    return _eltwise_kernel(arguments, operation, name, keep, options,
                           preamble, loop_prep, after_loop)


@cutools.context_dependent_memoize
def _reduce_kernel(dtype_out, neutral, reduce_expr, map_expr, arguments,
                   name, keep, options, preamble):
    return ReductionKernel(dtype_out, neutral, reduce_expr, map_expr, arguments,
                           name, keep, options, preamble)

def reduce(arguments, map_expr, reduce_expr, neutral, name,
           dtype_out=numpy.float32, keep=False, options=None, preamble=''):
    """Return memoized reduction kernel function."""
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
    """Get CUBLAS handle for current device."""

    global _cublas_handles

    device = Context.get_device()
    if device in _cublas_handles:
        return _cublas_handles[device]

    handle = cublas.cublasCreate()
    _cublas_handles[device] = handle
    return handle


class CumiscUser(object):
    def __init__(self, handle):
        self.handle = cumisc._global_cublas_handle
        self.tmp_handle = handle

    def __enter__(self):
        cumisc._global_cublas_handle = self.tmp_handle

    def __exit__(self, typ, value, traceback):
        cumisc._global_cublas_handle = self.handle


def using_cumisc(handle=None):
    """Temporarily use chainer's CUBLAS handle on scikits.cuda.misc."""
    if handle is None:
        handle = get_cublas_handle()
    return CumiscUser(handle)
