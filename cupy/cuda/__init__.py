import warnings

from cupy_backends.cuda.api import driver  # NOQA
from cupy_backends.cuda.api import runtime  # NOQA

from cupy._environment import get_cuda_path  # NOQA
from cupy._environment import get_nvcc_path  # NOQA
from cupy._environment import get_rocm_path  # NOQA
from cupy._environment import get_hipcc_path  # NOQA
from cupy._environment import get_cann_path # NOQA

from cupy.backends.backend.api.runtime import is_ascend

if not is_ascend():
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

# ---------------------------------------------------------------------------
# 兼容层：`cupy.cuda.<Class>` 是 CuPy 的历史公开路径，examples/ 与用户代码大量
# 使用它（`cupy.cuda.Stream` / `cupy.cuda.Device` / `cupy.cuda.MemoryPool` ...）。
# 本 fork 把设备 API 搬到了后端无关的 `cupy.xpu`，所以这里必须把**类和函数**也
# 重新导出：只 import 子模块（上面那些 `from cupy.xpu import stream`）会让
# `cupy.cuda.Stream` 变成 AttributeError —— 而且 `import cupy` 不会自动导入
# 子模块，所以 cupy/__init__.py 里还要有 `from cupy import cuda`。
# 只"新增名字、不删除"，CUDA/HIP 专有的部分按 is_ascend() 跳过（Ascend 上没有）。
# ---------------------------------------------------------------------------
from cupy.xpu import Device  # NOQA
from cupy.xpu import get_cublas_handle  # NOQA
from cupy.xpu import get_device_id  # NOQA
from cupy.xpu import alloc  # NOQA
from cupy.xpu import malloc_managed  # NOQA
from cupy.xpu import malloc_async  # NOQA
from cupy.xpu import BaseMemory  # NOQA
from cupy.xpu import ManagedMemory  # NOQA
from cupy.xpu import Memory  # NOQA
from cupy.xpu import MemoryAsync  # NOQA
from cupy.xpu import MemoryPointer  # NOQA
from cupy.xpu import MemoryPool  # NOQA
from cupy.xpu import PythonFunctionAllocator  # NOQA
from cupy.xpu import CFunctionAllocator  # NOQA
from cupy.xpu import set_allocator  # NOQA
from cupy.xpu import get_allocator  # NOQA
from cupy.xpu import UnownedMemory  # NOQA
from cupy.xpu import MemoryHook  # NOQA
from cupy.xpu import alloc_pinned_memory  # NOQA
from cupy.xpu import PinnedMemory  # NOQA
from cupy.xpu import PinnedMemoryPointer  # NOQA
from cupy.xpu import PinnedMemoryPool  # NOQA
from cupy.xpu import set_pinned_memory_allocator  # NOQA
from cupy.xpu import Event  # NOQA
from cupy.xpu import Stream  # NOQA
from cupy.xpu import ExternalStream  # NOQA
from cupy.xpu import get_current_stream  # NOQA
from cupy.xpu import get_elapsed_time  # NOQA
from cupy.xpu import using_allocator  # NOQA

if not is_ascend():
    # CUDA/HIP-only 的设备 API（Ascend 上没有对应的实现）
    from cupy.xpu import Function  # NOQA
    from cupy.xpu import Module  # NOQA
    from cupy.xpu import Graph  # NOQA
    from cupy.xpu import MemoryAsyncPool  # NOQA

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

