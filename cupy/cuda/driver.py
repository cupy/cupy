"""Thin wrapper of CUDA Driver API.

There are four differences compared to the original C API.

1. Not all functions are ported.
2. Errors are translated into CUDADriverError exceptions.
3. The 'cu' prefix of each API is omitted and the next character is set to
   lower case.
4. The resulting values are returned directly instead of references.

"""
import ctypes
import sys

import six

from cupy.cuda import internal

if 'win32' == sys.platform:
    _cuda = internal.load_library('nvcuda')
else:
    _cuda = internal.load_library('cuda')

###############################################################################
# Types
###############################################################################

Device = ctypes.c_int
Context = ctypes.c_void_p
Module = ctypes.c_void_p
Function = ctypes.c_void_p
Stream = ctypes.c_void_p
Event = ctypes.c_void_p

###############################################################################
# Error handling
###############################################################################

_cuda.cuGetErrorName.argtypes = (ctypes.c_int, ctypes.c_void_p)
_cuda.cuGetErrorString.argtypes = (ctypes.c_int, ctypes.c_void_p)


class CUDADriverError(RuntimeError):

    def __init__(self, status):
        self.status = status
        name = ctypes.c_char_p()
        msg = ctypes.c_char_p()
        _cuda.cuGetErrorName(status, ctypes.byref(name))
        _cuda.cuGetErrorString(status, ctypes.byref(msg))
        super(CUDADriverError, self).__init__(
            '%s: %s' % (name.value, msg.value))


def check_status(status):
    if status != 0:
        raise CUDADriverError(status)


###############################################################################
# Initialization
###############################################################################

_cuda.cuInit.argtypes = (ctypes.c_uint,)


def init():
    status = _cuda.cuInit(0)
    check_status(status)


_cuda.cuDriverGetVersion.argtypes = (ctypes.c_void_p,)


def driverGetVersion():
    version = ctypes.c_int()
    status = _cuda.cuDriverGetVersion(ctypes.byref(version))
    check_status(status)
    return version.value

###############################################################################
# Device and context operations
###############################################################################

_cuda.cuDeviceGet.argtypes = (ctypes.c_void_p, ctypes.c_int)


def deviceGet(device_id):
    device = Device()
    status = _cuda.cuDeviceGet(ctypes.byref(device), device_id)
    check_status(status)
    return device


_cuda.cuDeviceGetAttribute.argtypes = (ctypes.c_void_p, ctypes.c_int)


def deviceGetAttribute(attrib, device):
    ret = ctypes.c_int()
    status = _cuda.cuDeviceGetAttribute(ctypes.byref(ret), attrib, device)
    check_status(status)
    return ret


_cuda.cuDeviceGetCount.argtypes = (ctypes.c_void_p,)


def deviceGetCount():
    count = ctypes.c_int()
    status = _cuda.cuDeviceGetCount(ctypes.byref(count))
    check_status(status)
    return count.value


_cuda.cuDeviceTotalMem.argtypes = (ctypes.c_void_p, Device)


def deviceTotalMem(device):
    mem = ctypes.c_size_t()
    status = _cuda.cuDeviceTotalMem(ctypes.byref(mem), device)
    check_status(status)
    return mem.value


_cuda.cuCtxCreate_v2.argtypes = (ctypes.c_void_p, ctypes.c_uint, Device)


def ctxCreate(flag, device):
    ctx = Context()
    status = _cuda.cuCtxCreate_v2(ctypes.byref(ctx), flag, device)
    check_status(status)
    return ctx


_cuda.cuCtxDestroy_v2.argtypes = (ctypes.c_void_p,)


def ctxDestroy(ctx):
    status = _cuda.cuCtxDestroy_v2(ctx)
    check_status(status)


_cuda.cuCtxGetApiVersion.argtypes = (Context, ctypes.c_void_p)


def ctxGetApiVersion(ctx):
    version = ctypes.c_uint()
    status = _cuda.cuCtxGetApiVersion(ctx, ctypes.byref(version))
    check_status(status)
    return version.value


_cuda.cuCtxGetCurrent.argtypes = (ctypes.c_void_p,)


def ctxGetCurrent():
    ctx = Context()
    status = _cuda.cuCtxGetCurrent(ctypes.byref(ctx))
    check_status(status)
    return ctx


_cuda.cuCtxGetDevice.argtypes = (ctypes.c_void_p,)


def ctxGetDevice():
    device = Device()
    status = _cuda.cuCtxGetDevice(ctypes.byref(device))
    check_status(status)
    return device


_cuda.cuCtxPopCurrent_v2.argtypes = (ctypes.c_void_p,)


def ctxPopCurrent():
    ctx = Context()
    status = _cuda.cuCtxPopCurrent_v2(ctypes.byref(ctx))
    check_status(status)
    return ctx


_cuda.cuCtxPushCurrent_v2.argtypes = (Context,)


def ctxPushCurrent(ctx):
    status = _cuda.cuCtxPushCurrent_v2(ctx)
    check_status(status)


_cuda.cuCtxSetCurrent.argtypes = (Context,)


def ctxSetCurrent(ctx):
    status = _cuda.cuCtxSetCurrent(ctx)
    check_status(status)


def ctxSynchronize():
    status = _cuda.cuCtxSynchronize()
    check_status(status)


###############################################################################
# Module load and kernel execution
###############################################################################

_cuda.cuModuleLoad.argtypes = (ctypes.c_void_p, ctypes.c_char_p)


def moduleLoad(filename):
    module = Module()
    status = _cuda.cuModuleLoad(ctypes.byref(module), filename)
    check_status(status)
    return module


_cuda.cuModuleLoadData.argtypes = (ctypes.c_void_p, ctypes.c_char_p)


def moduleLoadData(image):
    module = Module()
    status = _cuda.cuModuleLoadData(ctypes.byref(module), image)
    check_status(status)
    return module


_cuda.cuModuleUnload.argtypes = (Module,)


def moduleUnload(module):
    status = _cuda.cuModuleUnload(module)
    check_status(status)


_cuda.cuModuleGetFunction.argtypes = (ctypes.c_void_p, Module, ctypes.c_char_p)


def moduleGetFunction(module, funcname):
    func = Function()
    if isinstance(funcname, six.text_type):
        funcname = funcname.encode('utf-8')
    status = _cuda.cuModuleGetFunction(ctypes.byref(func), module, funcname)
    check_status(status)
    return func


_cuda.cuModuleGetGlobal_v2.argtypes = (
    ctypes.c_void_p, Module, ctypes.c_char_p)


def moduleGetGlobal(module, varname):
    var = ctypes.c_void_p()
    status = _cuda.cuModuleGetGlobal_v2(ctypes.byref(var), module, varname)
    check_status(status)
    return var


_cuda.cuLaunchKernel.argtypes = (
    Function, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, Stream, ctypes.c_void_p,
    ctypes.c_void_p)


def launchKernel(f, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x,
                 block_dim_y, block_dim_z, shared_mem_bytes, stream,
                 kernel_params, extra):
    status = _cuda.cuLaunchKernel(
        f, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
        block_dim_z, shared_mem_bytes, stream, kernel_params, extra)
    check_status(status)


###############################################################################
# Memory management
###############################################################################

_cuda.cuMemAlloc_v2.argtypes = (ctypes.c_void_p, ctypes.c_size_t)


def memAlloc(size):
    ptr = ctypes.c_void_p()
    status = _cuda.cuMemAlloc_v2(ctypes.byref(ptr), size)
    check_status(status)
    return ptr


_cuda.cuMemFree_v2.argtypes = (ctypes.c_void_p,)


def memFree(ptr):
    status = _cuda.cuMemFree_v2(ptr)
    check_status(status)


_cuda.cuMemGetInfo_v2.argtypes = (ctypes.c_void_p, ctypes.c_void_p)


def memGetinfo():
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    status = _cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
    check_status(status)
    return free.value, total.value


_cuda.cuMemcpy.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)


def memcpy(dst, src, size):
    status = _cuda.cuMemcpy(dst, src, size)
    check_status(status)


_cuda.cuMemcpyAsync.argtypes = (ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_size_t, Stream)


def memcpyAsync(dst, src, size, stream):
    status = _cuda.cuMemcpyAsync(dst, src, size, stream)
    check_status(status)


_cuda.cuMemcpyDtoD_v2.argtypes = (ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t)


def memcpyDtoD(dst, src, size):
    status = _cuda.cuMemcpyDtoD_v2(dst, src, size)
    check_status(status)


_cuda.cuMemcpyDtoDAsync_v2.argtypes = (ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_size_t, Stream)


def memcpyDtoDAsync(dst, src, size, stream):
    status = _cuda.cuMemcpyDtoDAsync_v2(dst, src, size, stream)
    check_status(status)


_cuda.cuMemcpyDtoH_v2.argtypes = (ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t)


def memcpyDtoH(dst, src, size):
    status = _cuda.cuMemcpyDtoH_v2(dst, src, size)
    check_status(status)


_cuda.cuMemcpyDtoHAsync_v2.argtypes = (ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_size_t, Stream)


def memcpyDtoHAsync(dst, src, size, stream):
    status = _cuda.cuMemcpyDtoHAsync_v2(dst, src, size, stream)
    check_status(status)


_cuda.cuMemcpyHtoD_v2.argtypes = (ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t)


def memcpyHtoD(dst, src, size):
    status = _cuda.cuMemcpyHtoD_v2(dst, src, size)
    check_status(status)


_cuda.cuMemcpyHtoDAsync_v2.argtypes = (ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_size_t, Stream)


def memcpyHtoDAsync(dst, src, size, stream):
    status = _cuda.cuMemcpyHtoDAsync_v2(dst, src, size, stream)
    check_status(status)


_cuda.cuMemcpyPeer.argtypes = (ctypes.c_void_p, Context, ctypes.c_void_p,
                               Context, ctypes.c_size_t)


def memcpyPeer(dst, dst_ctx, src, src_ctx, size):
    status = _cuda.cuMemcpyPeer(dst, dst_ctx, src, src_ctx, size)
    check_status(status)


_cuda.cuMemcpyPeerAsync.argtypes = (ctypes.c_void_p, Context, ctypes.c_void_p,
                                    Context, ctypes.c_size_t, Stream)


def memcpyPeerAsync(dst, dst_ctx, src, src_ctx, size, stream):
    status = _cuda.cuMemcpyPeerAsync(dst, dst_ctx, src, src_ctx, size, stream)
    check_status(status)


_cuda.cuMemsetD32_v2.argtypes = (
    ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t)


def memsetD32(ptr, value, size):
    status = _cuda.cuMemsetD32_v2(ptr, value, size)
    check_status(status)


_cuda.cuMemsetD32Async.argtypes = (ctypes.c_void_p, ctypes.c_uint,
                                   ctypes.c_size_t, Stream)


def memsetD32Async(ptr, value, size, stream):
    status = _cuda.cuMemsetD32Async(ptr, value, size, stream)
    check_status(status)


_cuda.cuPointerGetAttribute.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_void_p)


def pointerGetAttribute(attribute, ptr):
    assert attribute == 0  # Currently only context query is supported

    ctx = Context()
    status = _cuda.cuPointerGetAttribute(ctypes.byref(ctx), attribute, ptr)
    check_status(status)
    return ctx


###############################################################################
# Stream and Event
###############################################################################

_cuda.cuStreamCreate.argtypes = (ctypes.c_void_p, ctypes.c_uint)


def streamCreate(flag=0):
    stream = Stream()
    status = _cuda.cuStreamCreate(ctypes.byref(stream), flag)
    check_status(status)
    return stream


_cuda.cuStreamDestroy_v2.argtypes = (Stream,)


def streamDestroy(stream):
    status = _cuda.cuStreamDestroy_v2(stream)
    check_status(status)


_cuda.cuStreamSynchronize.argtypes = (Stream,)


def streamSynchronize(stream):
    status = _cuda.cuStreamSynchronize(stream)
    check_status(status)


StreamCallback = ctypes.CFUNCTYPE(Stream, ctypes.c_int, ctypes.c_void_p)
_cuda.cuStreamAddCallback.argtypes = (Stream, StreamCallback, ctypes.c_void_p,
                                      ctypes.c_uint)


def streamAddCallback(stream, callback, arg, flags=0):
    status = _cuda.cuStreamAddCallback(stream, StreamCallback(callback),
                                       ctypes.byref(arg), flags)
    check_status(status)


EVENT_DEFAULT = 0
EVENT_BLOCKING_SYNC = 1
EVENT_DISABLE_TIMING = 2
EVENT_INTERPROCESS = 4
_cuda.cuEventCreate.argtypes = (ctypes.c_void_p, ctypes.c_uint)


def eventCreate(flag):
    event = Event()
    status = _cuda.cuEventCreate(ctypes.byref(event), flag)
    check_status(status)
    return event


_cuda.cuEventDestroy_v2.argtypes = (Event,)


def eventDestroy(event):
    status = _cuda.cuEventDestroy_v2(event)
    check_status(status)


_cuda.cuEventRecord.argtypes = (Event, Stream)


def eventRecord(event, stream):
    status = _cuda.cuEventRecord(event, stream)
    check_status(status)


_cuda.cuEventSynchronize.argtypes = (Event,)


def eventSynchronize(event):
    status = _cuda.cuEventSynchronize(event)
    check_status(status)
