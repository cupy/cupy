import atexit
import collections

import six

from cupy.cuda import cublas
from cupy.cuda import runtime


class Device(object):
    _cublas_handles = {}

    def __init__(self, device_id=-1):
        if device_id < 0:
            device_id = runtime.getDevice()
        self.id = device_id

    def use(self):
        runtime.setDevice(self.id)

    @staticmethod
    def synchronize():
        runtime.deviceSynchronize()

    @property
    def compute_capability(self):
        major = runtime.deviceGetAttribute(75, self.id)
        minor = runtime.deviceGetAttribute(76, self.id)
        return '%d%d' % (major, minor)

    @property
    def cublas_handle(self):
        handle = self._cublas_handles.get(self.id, None)
        if handle is None:
            with DeviceUser(self):
                handle = cublas.create()
                self._cublas_handles[self.id] = handle
        return handle

    def __eq__(self, other):
        if not isinstance(other, Device):
            return False
        return self.id == other.id

    def __ne__(self, other):
        return not (self == other)


def from_pointer(ptr):
    attrs = runtime.pointerGetAttributes(ptr)
    return Device(attrs.device)


class DeviceUser(object):

    def __init__(self, device):
        self.prev_device = Device()
        self.cur_device = device

    def __enter__(self):
        self.cur_device.use()

    def __exit__(self, typ, val, trace):
        self.prev_device.use()


@atexit.register
def destroy_cublas_handles():
    for handle in six.itervalues(Device._cublas_handles):
        cublas.destroy(handle)
    Device._cublas_handles = {}


_memoized_funcs = []


def memoize(f):
    def func(*args, **kwargs):
        # TODO(okuta): Improve keyword arguments.
        global _memoized_funcs

        if not hasattr(f, '_cupy_dev_memo'):
            _memoized_funcs.append(f)
            f._cupy_dev_memo = collections.defaultdict(dict)

        memo = f._cupy_dev_memo[Device().id]
        arg_key = (args, frozenset(kwargs.items()))
        result = memo.get(arg_key, None)
        if result is None:
            result = f(*args, **kwargs)
            memo[arg_key] = result
        return result

    return func


@atexit.register
def clear_device_dependent_memo():
    global _memoized_funcs
    for func in _memoized_funcs:
        del func._cupy_dev_memo
    _memoized_funcs = []
