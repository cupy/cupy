from cupy_backends.cuda.api import driver  # NOQA
from cupy_backends.cuda.api import runtime  # NOQA

from cupy._environment import get_cuda_path  # NOQA
from cupy._environment import get_nvcc_path  # NOQA
from cupy._environment import get_rocm_path  # NOQA
from cupy._environment import get_hipcc_path  # NOQA
from cupy._environment import get_cann_path # NOQA

#from cupy_backends.cuda.runtime import is_ascend as _is_ascend

_is_ascend = True # TODO: ASCEND get from env?
if not _is_ascend:
    from cupy.xpu import compiler  # NOQA
    from cupy.xpu import texture  # NOQA
from cupy.xpu import function  # NOQA
from cupy.xpu import device  # NOQA
from cupy.xpu import memory  # NOQA
from cupy.xpu import memory_hook  # NOQA
from cupy.xpu import memory_hooks  # NOQA
from cupy.xpu import pinned_memory  # NOQA
from cupy.xpu import profiler  # NOQA
from cupy.xpu import stream  # NOQA

import cupy as _cupy
_available = None


class _UnavailableModule:
    available = False

    def __init__(self, name):
        self.__name__ = name

try:
    from cupy_backends.cuda.libs import nvrtc  # NOQA
except ImportError:
    cub = _UnavailableModule('cupy.cuda.nvrtc')

try:
    from cupy.cuda import cub  # NOQA
except ImportError:
    cub = _UnavailableModule('cupy.cuda.cub')

try:
    from cupy_backends.cuda.libs import nvtx  # NOQA
except ImportError:
    nvtx = _UnavailableModule('cupy.cuda.nvtx')

try:
    from cupy.cuda import thrust  # NOQA
except ImportError:
    thrust = _UnavailableModule('cupy.cuda.thrust')


def __getattr__(key):
    if key == 'cusolver':
        from cupy_backends.cuda.libs import cusolver
        _cupy.cuda.cusolver = cusolver
        return cusolver
    elif key == 'cusparse':
        from cupy_backends.cuda.libs import cusparse
        _cupy.cuda.cusparse = cusparse
        return cusparse
    elif key == 'curand':
        from cupy_backends.cuda.libs import curand
        _cupy.cuda.curand = curand
        return curand
    elif key == 'cublas':
        from cupy_backends.cuda.libs import cublas
        _cupy.cuda.cublas = cublas
        return cublas
    elif key == 'jitify':
        if not runtime.is_hip and driver.get_build_version() > 0:
            import cupy.cuda.jitify as jitify
        else:
            jitify = _UnavailableModule('cupy.cuda.jitify')
        _cupy.cuda.jitify = jitify
        return jitify

    # `nvtx_enabled` flags are kept for backward compatibility with Chainer.
    # Note: module-level getattr only runs on Python 3.7+.
    for mod in [nvtx]:
        flag = '{}_enabled'.format(mod.__name__.split('.')[-1])
        if key == flag:
            warnings.warn('''
cupy.cuda.{} has been deprecated in CuPy v8 and will be removed in the future release.
Use {}.available instead.
                '''.format(flag, mod.__name__), DeprecationWarning)  # NOQA
            return not isinstance(mod, _UnavailableModule)

    raise AttributeError(
        "module '{}' has no attribute '{}'".format(__name__, key))


def is_available():
    global _available
    if _available is None:
        _available = False
        try:
            _available = runtime.getDeviceCount() > 0
        except Exception as e:
            if (not runtime.is_hip and e.args[0] !=
                    'cudaErrorNoDevice: no CUDA-capable device is detected'):
                raise
            elif runtime.is_hip and 'hipErrorNoDevice' not in e.args[0]:
                raise
    return _available


def get_local_runtime_version() -> int:
    """
    Returns the version of the CUDA Runtime installed in the environment.

    Unlike :func:`cupy.cuda.runtime.runtimeGetVersion`, which returns the
    CUDA Runtime version statically linked to CuPy, this function returns the
    version retrieved from the shared library installed on the host.
    Use this method to probe the CUDA Runtime version installed in the
    environment.
    """
    return runtime._getLocalRuntimeVersion()

