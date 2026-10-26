"""Ascend 下 ``cupy._core._accelerator`` 的兼容桩（stub）。

CUDA 侧的机制（见 ``cupy/_core/_gpu/_accelerator.pyx``）：
    启动时从 ``CUPY_ACCELERATORS`` 环境变量（默认 ``'cub'``）或
    ``set_*_accelerators()`` 读取"加速器优先级列表"，reduce/scan/
    tensor-contraction 例程（``_routines_math.pyx``、
    ``_routines_statistics.pyx``、linalg contraction 等）按列表逐个尝试
    NVIDIA 官方优化库实现（CUB DeviceReduce/DeviceScan、cuTENSOR、
    cuTENSORNET），命中且 dtype/shape 兼容则用之，否则回退 CuPy 内置
    ElementwiseKernel/ReductionKernel。

Ascend 后端不需要该机制：aclnn 算子（``ascend_*``，见
``cupy/backends/ascend/``）本身就是"厂商加速库"路径，在 dispatch 层
默认接管全部例程，没有可选的二级加速器，故本模块提供空实现。

作用仅是保持 ``cupy._core`` 顶层名字与 CUDA 路径一致（否则共享代码里
``cupy._core.set_elementwise_accelerators`` 之类的引用会在 Ascend 上
AttributeError），set_* 为 no-op，get_* 返回空列表。
"""

# 与 cupy/_core/_accelerator.pxd 的 cpdef enum 保持同值，供兼容代码引用
ACCELERATOR_CUB = 1
ACCELERATOR_CUTENSOR = 2
ACCELERATOR_CUTENSORNET = 3


def set_elementwise_accelerators(accelerators):
    pass


def set_reduction_accelerators(accelerators):
    pass


def set_routine_accelerators(accelerators):
    pass


def get_elementwise_accelerators():
    return []


def get_reduction_accelerators():
    return []


def get_routine_accelerators():
    return []
