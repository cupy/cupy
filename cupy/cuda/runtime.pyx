"""Thin wrapper of CUDA Runtime API.

There are four differences compared to the original C API.

1. Not all functions are ported.
2. Errors are translated into CUDARuntimeError exceptions.
3. The 'cuda' prefix of each API is omitted and the next character is set to
   lower case.
4. The resulting values are returned directly instead of references.

"""
import ctypes
import sys

from cupy.cuda import internal

if 'win32' == sys.platform:
    _cudart = internal.load_library(
        internal.get_windows_cuda_library_names('cudart'))
else:
    _cudart = internal.load_library('cudart')

###############################################################################
# Types
###############################################################################

Device = ctypes.c_int
Function = ctypes.c_void_p
Stream = ctypes.c_void_p
Event = ctypes.c_void_p

memcpyHostToHost = 0
memcpyHostToDevice = 1
memcpyDeviceToHost = 2
memcpyDeviceToDevice = 3
memcpyDefault = 4


###############################################################################
# Error handling
###############################################################################

_cudart.cudaGetErrorName.restype = ctypes.c_char_p
_cudart.cudaGetErrorName.argtypes = (ctypes.c_int,)
_cudart.cudaGetErrorString.restype = ctypes.c_char_p
_cudart.cudaGetErrorString.argtypes = (ctypes.c_int,)


class CUDARuntimeError(RuntimeError):

    def __init__(self, status):
        self.status = status
        name = _cudart.cudaGetErrorName(status)
        msg = _cudart.cudaGetErrorString(status)
        super(CUDARuntimeError, self).__init__('%s: %s' % (name, msg))


def check_status(status):
    if status != 0:
        raise CUDARuntimeError(status)


###############################################################################
# Initialization
###############################################################################

_cudart.cudaDriverGetVersion.argtypes = (ctypes.c_void_p,)


def driverGetVersion():
    version = ctypes.c_int()
    status = _cudart.cudaDriverGetVersion(ctypes.byref(version))
    check_status(status)
    return version.value


###############################################################################
# Device and context operations
###############################################################################

_cudart.cudaGetDevice.argtypes = (ctypes.POINTER(ctypes.c_int),)


def getDevice():
    device = Device()
    status = _cudart.cudaGetDevice(ctypes.byref(device))
    if status != 0:
        check_status(status)
    return device.value


_cudart.cudaDeviceGetAttribute.argtypes = (ctypes.POINTER(ctypes.c_int),
                                           ctypes.c_int, Device)


def deviceGetAttribute(attrib, device):
    ret = ctypes.c_int()
    status = _cudart.cudaDeviceGetAttribute(ctypes.byref(ret), attrib, device)
    check_status(status)
    return ret.value


_cudart.cudaGetDeviceCount.argtypes = (ctypes.POINTER(ctypes.c_int),)


def getDeviceCount():
    count = ctypes.c_int()
    status = _cudart.cudaGetDeviceCount(ctypes.byref(count))
    check_status(status)
    return count.value


_cudart.cudaSetDevice.argtypes = (Device,)


def setDevice(device):
    status = _cudart.cudaSetDevice(Device(device))
    check_status(status)


def deviceSynchronize():
    status = _cudart.cudaDeviceSynchronize()
    check_status(status)


###############################################################################
# Memory management
###############################################################################

_cudart.cudaMalloc.argtypes = (ctypes.c_void_p, ctypes.c_size_t)


def malloc(size):
    ptr = ctypes.c_void_p()
    status = _cudart.cudaMalloc(ctypes.byref(ptr), size)
    check_status(status)
    return ptr


_cudart.cudaFree.argtypes = (ctypes.c_void_p,)


def free(ptr):
    status = _cudart.cudaFree(ptr)
    check_status(status)


_cudart.cudaMemGetInfo.argtypes = (ctypes.POINTER(ctypes.c_int),
                                   ctypes.POINTER(ctypes.c_int))


def memGetInfo():
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    status = _cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    check_status(status)
    return free.value, total.value


_cudart.cudaMemcpy.argtypes = (ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_size_t, ctypes.c_int)


def memcpy(dst, src, size, kind):
    status = _cudart.cudaMemcpy(dst, src, size, kind)
    check_status(status)


_cudart.cudaMemcpyAsync.argtypes = (ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_size_t, ctypes.c_int, Stream)


def memcpyAsync(dst, src, size, kind, stream):
    status = _cudart.cudaMemcpyAsync(dst, src, size, kind, stream)
    check_status(status)


_cudart.cudaMemcpyPeer.argtypes = (ctypes.c_void_p, Device,
                                   ctypes.c_void_p, Device, ctypes.c_size_t)


def memcpyPeer(dst, dstDevice, src, srcDevice, size):
    status = _cudart.cudaMemcpyPeer(dst, dstDevice, src, srcDevice, size)
    check_status(status)


_cudart.cudaMemcpyPeerAsync.argtypes = (ctypes.c_void_p, Device,
                                        ctypes.c_void_p, Device,
                                        ctypes.c_size_t, Stream)


def cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, size, stream):
    status = _cudart.cudaMemcpyPeer(dst, dstDevice,
                                    src, srcDevice, size, stream)
    check_status(status)


_cudart.cudaMemset.argtypes = (ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t)


def memset(ptr, value, size):
    status = _cudart.cudaMemset(ptr, value, size)
    check_status(status)


_cudart.cudaMemsetAsync.argtypes = (ctypes.c_void_p, ctypes.c_uint,
                                    ctypes.c_size_t, Stream)


def memsetAsync(ptr, value, size, stream):
    status = _cudart.cudaMemsetAsync(ptr, value, size, stream)
    check_status(status)


cudaMemoryTypeHost = 1
cudaMemoryTypeDevice = 2


class cudaPointerAttributes(ctypes.Structure):
    _fields_ = (
        ('memoryType', ctypes.c_int),
        ('device', ctypes.c_int),
        ('devicePointer', ctypes.c_void_p),
        ('hostPointer', ctypes.c_void_p),
        ('isManaged', ctypes.c_int),
    )

_cudart.cudaPointerGetAttributes.argtypes = (
    ctypes.POINTER(cudaPointerAttributes), ctypes.c_void_p)


def pointerGetAttributes(ptr):
    attrs = cudaPointerAttributes()
    status = _cudart.cudaPointerGetAttributes(ctypes.byref(attrs), ptr)
    check_status(status)
    return attrs


###############################################################################
# Stream and Event
###############################################################################

_cudart.cudaStreamCreate.argtypes = (ctypes.POINTER(Stream),)


def streamCreate():
    stream = Stream()
    status = _cudart.cudaStreamCreate(ctypes.byref(stream))
    check_status(status)
    return stream


streamDefault = 0
streamNonBlocking = 1
_cudart.cudaStreamCreateWithFlags.argtypes = (ctypes.POINTER(Stream),
                                              ctypes.c_int)


def streamCreateWithFlags(flags):
    stream = Stream()
    status = _cudart.cudaStreamCreateWithFlags(ctypes.byref(stream), flags)
    check_status(status)
    return stream


_cudart.cudaStreamDestroy.argtypes = (Stream,)


def streamDestroy(stream):
    status = _cudart.cudaStreamDestroy(stream)
    check_status(status)


_cudart.cudaStreamSynchronize.argtypes = (Stream,)


def streamSynchronize(stream):
    status = _cudart.cudaStreamSynchronize(stream)
    check_status(status)


StreamCallback = ctypes.CFUNCTYPE(Stream, ctypes.c_int, ctypes.c_void_p)
_cudart.cudaStreamAddCallback.argtypes = (Stream, StreamCallback,
                                          ctypes.c_void_p, ctypes.c_uint)


def streamAddCallback(stream, callback, arg, flags=0):
    status = _cudart.cudaStreamAddCallback(stream, StreamCallback(callback),
                                           ctypes.byref(arg), flags)
    check_status(status)


_cudart.cudaStreamQuery.argtypes = (Stream,)


def streamQuery(stream):
    return _cudart.cudaStreamQuery(stream)


_cudart.cudaStreamWaitEvent.argtypes = (Stream, Event, ctypes.c_uint)


def streamWaitEvent(stream, event, flags=0):
    status = _cudart.cudaStreamWaitEvent(stream, event, flags)
    check_status(status)


_cudart.cudaEventCreate.argtypes = (ctypes.POINTER(Event),)


def eventCreate():
    event = Event()
    status = _cudart.cudaEventCreate(ctypes.byref(event))
    check_status(status)
    return event


eventDefault = 0
eventBlockingSync = 1
eventDisableTiming = 2
eventInterprocess = 4
_cudart.cudaEventCreateWithFlags.argtypes = (ctypes.POINTER(Event),
                                             ctypes.c_int)


def eventCreateWithFlags(flags):
    event = Event()
    status = _cudart.cudaEventCreateWithFlags(ctypes.byref(event), flags)
    check_status(status)
    return event


_cudart.cudaEventDestroy.argtypes = (Event,)


def eventDestroy(event):
    status = _cudart.cudaEventDestroy(event)
    check_status(status)


_cudart.cudaEventElapsedTime.argtypes = (ctypes.POINTER(ctypes.c_float),
                                         Event, Event)


def eventElapsedTime(start, end):
    ms = ctypes.c_float()
    status = _cudart.cudaEventElapsedTime(ctypes.byref(ms), start, end)
    check_status(status)
    return ms.value


_cudart.cudaEventQuery.argtypes = (Event,)


def eventQuery(event):
    return _cudart.cudaEventQuery(event)


_cudart.cudaEventRecord.argtypes = (Event, Stream)


def eventRecord(event, stream):
    status = _cudart.cudaEventRecord(event, stream)
    check_status(status)


_cudart.cudaEventSynchronize.argtypes = (Event,)


def eventSynchronize(event):
    status = _cudart.cudaEventSynchronize(event)
    check_status(status)
