"""Device, context and memory management on PyCUDA."""

import atexit, os
import numpy
import pycuda.driver as drv
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction   import ReductionKernel
from pycuda import gpuarray
from pycuda import curandom
import pycuda.tools as cutools
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
_pid      = None
_pool     = None

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
    global generator, mem_alloc, _contexts, _pid, _pool

    pid = os.getpid()
    if _pid == pid:  # already initialized
        return

    drv.init()

    if device is None:  # use default device
        context = cutools.make_default_context()
    else:
        device   = Device(device)
        context = device.make_context()
    _contexts = {Context.get_device(): context}

    _pool     = cutools.DeviceMemoryPool()
    mem_alloc = _pool.allocate

    generator = curandom.XORWOWRandomNumberGenerator(seed_getter=_seed_getter)

    _pid      = pid  # mark as initialized
    atexit.register(shutdown)

    cumisc.init(mem_alloc)


def shutdown():
    """Finalize CUDA global state."""

    global mem_alloc, _contexts, _pid, _pool

    pid = os.getpid()
    if _pid != pid:  # not initialized
        return

    cumisc.shutdown()

    mem_alloc = None
    _pool     = None

    for ctx in _contexts.itervalues():
        ctx.detach()
    _contexts = {}
    _pid      = None  # mark as uninitialized


def use_device(device):
    """Switch context to use given device."""

    if not isinstance(device, drv.Device):
        device = drv.Device(device)

    drv.Context.pop()
    if device not in _contexts:
        _contexts[device] = device.make_context()
    else:
        _contexts[device].push()


def free_pool():
    """Free memory allocations held by memory pool."""
    if _pool is not None:
        _pool.free_held()


# ------------------------------------------------------------------------------
# GPUArray allocation and copy
# ------------------------------------------------------------------------------
def to_gpu(array):
    """Copy array to the current device."""
    if isinstance(array, GPUArray):
        return array
    return gpuarray.to_gpu(array, allocator=mem_alloc)


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


def empty(shape, dtype):
    """Create an uninitialized GPUArray."""
    return gpuarray.empty(shape, dtype, allocator=mem_alloc)


def full(shape, fill_value, dtype, stream=None):
    """Create constant-filled GPUArray."""
    array = empty(shape, dtype)
    array.fill(fill_value, stream=stream)
    return array


def zeros(shape, dtype, stream=None):
    """Create zero-filled GPUArray."""
    return full(shape, 0, dtype, stream=stream)


empty_like = gpuarray.empty_like


def full_like(array, fill_value, stream=None):
    """Create a constant-filled GPUArray like ``array``."""
    array = empty_like(array)
    array.fill(fill_value, stream=stream)
    return array


def zeros_like(array, stream=None):
    """Create a zero-filled GPUArray like ``array``."""
    return full_like(array, 0, stream=stream)
    

def copy(array, out=None):
    """Copy GPUArray in default stream."""
    if out is None:
        out = empty_like(array)
    drv.memcpy_dtod(out.ptr, array.ptr, out.nbytes)
    return out


def copy_async(array, out=None, stream=None):
    """Copy GPUArray asynchronously."""
    if out is None:
        out = empty_like(array)
    drv.memcpy_dtod_async(out.ptr, array.ptr, out.nbytes, stream=stream)
    return out


def copy_peer(array, out_device, src_device, out=None):
    """Copy GPUArray over devices synchronously."""
    use_device(out_device)
    if out is None:
        out = empty_like(array)
    drv.memcpy_peer(out.ptr, array.ptr, out.nbytes, out_device, src_device)
    drv.Context.pop()
    return out


def copy_peer_async(array, out_device, src_device, out=None, stream=None):
    """Copy GPUArray over devices asynchronously."""
    use_device(out_device)
    if out is None:
        out = empty_like(array)
    drv.memcpy_peer_async(out.ptr, array.ptr, out.nbytes, out_device, src_device,
                           stream=stream)
    drv.Context.pop()
    return out


def copy_peer_async(array, out=None, device=None, stream=None):
    """Copy GPUArray over devices asynchronously.

    ``array`` must reside at the current device.

    """
    cur_device = drv.Context.get_device()
    if out is None:
        assert device is not None
        use_device(device)
        out = empty(array.shape, array.dtype)


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

def reduce(arguments, map_expr, reduce_expr, neutral, name, dtype_out,
           keep=False, options=None, preamble=''):
    """Return memoized reduction kernel function."""
    kern = _reduce_kernel(dtype_out, neutral, reduce_expr, map_expr, arguments,
                          name, keep, options, preamble)
    def call_kern(*args, **kwargs):
        kwargs['allocator'] = mem_alloc
        return kern(*args, **kwargs)
    return call_kern
