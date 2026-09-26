"""Ascend(NPU) CPU fallback 基础设施：把 NPU 上没有实现的算子搬到 host 用 NumPy 算。

背景
----
CANN 的 aclnn / ops-blas 覆盖不了全部 NumPy 线性代数接口。对照
``~/repos/ops-blas/docs/zh/api_list.md``：BLAS L1/L2/L3 加上 LAPACK 风格的**批量**接口
只提供 ``getrf / getri / getrs / geqrf / gels / matinv``（且本项目尚未接入 ops-blas），
**没有** ``det / slogdet / eig / eigvals / eigh / eigvalsh / cholesky`` 这类行列式、
特征值与 Cholesky 接口。

这类算子的第三条路（前两条是 aclnn 包装与 AscendC 自定义内核）就是本模块：

    np_in  = to_numpy(device_in)          # cupy.ndarray -> numpy.ndarray  (D2H)
    np_out = numpy.linalg.det(np_in)      # host 计算
    out    = to_device(np_out)            # numpy.ndarray -> cupy.ndarray (H2D)

:func:`call` 把这三步串起来并按 :data:`FALLBACKS` 注册表派发；
:func:`run` 供不在注册表里的个别算子直接使用（传 NumPy 函数即可）。

只用 XPU 中性的 array API
-------------------------
D2H / H2D 与设备端数组创建**不直接碰 ``cupy.cuda.*`` / ``cupy.xpu.device``**，
而是走中性层（见 Roadmap.md §5「XPU API 中性化重构」）：

* D2H：``cupy.asnumpy(a)`` —— 内部 ``ndarray.get()`` -> ``cupy.xpu`` 抽象
* H2D：``cupy.asarray(np_array)`` —— 内部 ``cupy._core._routines_creation`` -> ``cupy.xpu`` 抽象

于是同一个 fallback 在 CUDA / ROCm / Ascend 后端都能工作（只是别的后端不需要它）。

约定与已知偏差
--------------
1. **只在 :func:`active` 为真时开放**（Ascend 后端、且未设
   ``CUPY_ASCEND_DISABLE_CPU_FALLBACK=1``）。否则 :func:`call` 直接
   ``NotImplementedError`` —— 避免在 CUDA 后端悄悄替换掉本来有的 cuSOLVER 实现。
2. **dtype 语义以 NumPy 为准**：NumPy 的 ``linalg`` 不接受 ``float16``（抛 ``TypeError``），
   而 CuPy 的 ``eigh / eigvalsh`` 允许 fp16（按 fp32 计算）。在 fallback 路径上 fp16
   **响亮报错**，不静默降精度。
3. **性能**：每次调用两次 D2H/H2D 拷贝，适合低频、矩阵规模不大的 API
   （``det`` / ``eigvals`` 这类）；大矩阵或批量场景应等 ops-blas 的 LAPACK 批量接口接入。
4. 返回值里的 NumPy namedtuple（``numpy.linalg.EighResult`` 等）会保留结构，
   但各字段已换成 ``cupy.ndarray``。

验证等级：本机无 NPU 时只能到 L3（import + 接线 + 依赖对账）；数值正确性待 910B。
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import numpy

__all__ = [
    'DISABLE_ENV',
    'FALLBACKS',
    'active',
    'available',
    'call',
    'reset_cache',
    'run',
    'to_device',
    'to_numpy',
]

#: 设为 ``1`` 时关闭全部 CPU fallback（算子会退回"显式失败"）。
DISABLE_ENV = 'CUPY_ASCEND_DISABLE_CPU_FALLBACK'

#: 算子名 -> host 端 NumPy 实现。
#: 命名约定：``<模块>.<公开名>``（``linalg.`` 前缀对应 ``cupy.linalg.*``）。
#: ``tools/`` / 测试可以据此对账"接线里调用的名字一定在注册表里"。
FALLBACKS: Dict[str, Callable[..., Any]] = {
    'linalg.cholesky': numpy.linalg.cholesky,
    'linalg.det': numpy.linalg.det,
    'linalg.slogdet': numpy.linalg.slogdet,
    'linalg.eig': numpy.linalg.eig,
    'linalg.eigvals': numpy.linalg.eigvals,
    'linalg.eigh': numpy.linalg.eigh,
    'linalg.eigvalsh': numpy.linalg.eigvalsh,
    # percentile/quantile 的 'linear'/'midpoint' 插值走内联
    # cupy_percentile_weightnening ElementwiseKernel（raw CUDA body），
    # Ascend 后端无法执行，整个 quantile 改在 host 端用 NumPy 算
    # （cupy._statistics.order._quantile_unchecked 接线）。
    'statistics.quantile': numpy.quantile,
}

_ASCEND: Optional[bool] = None


# ---------------------------------------------------------------------------
# 后端判断
# ---------------------------------------------------------------------------
def active() -> bool:
    """当前是否允许 CPU fallback（Ascend 后端 + 未被环境变量关闭）。"""
    global _ASCEND
    if _ASCEND is None:
        try:
            from cupy.backends.backend import is_ascend
            _ASCEND = bool(is_ascend)
        except Exception:      # pragma: no cover - 非 ascend 环境
            _ASCEND = False
    if not _ASCEND:
        return False
    return os.getenv(DISABLE_ENV, '0') not in ('1', 'true', 'True')


def reset_cache() -> None:
    """清掉 :func:`active` 的后端判断缓存（环境变量不缓存）。"""
    global _ASCEND
    _ASCEND = None


def available(name: str) -> bool:
    """``name`` 是否在 :data:`FALLBACKS` 里。"""
    return name in FALLBACKS


def _cupy():
    import cupy
    return cupy


def _check_active(name: str) -> Callable[..., Any]:
    if not active():
        raise NotImplementedError(
            'CPU fallback 未启用（后端非 Ascend 或设置了 {}=1），'
            '无法执行 {}'.format(DISABLE_ENV, name))
    func = FALLBACKS.get(name)
    if func is None:
        raise KeyError(
            '未知的 CPU fallback 算子 {!r}（已注册: {}）'
            .format(name, ', '.join(sorted(FALLBACKS))))
    return func


# ---------------------------------------------------------------------------
# D2H / H2D
# ---------------------------------------------------------------------------
def to_numpy(value: Any) -> Any:
    """把设备端数组搬到 host；非 ``cupy.ndarray`` 原样返回。

    走 ``cupy.asnumpy``（= ``ndarray.get()``）而不是 ``cupy.xpu``/``cupy.cuda``
    的底层接口，保持后端中性。
    """
    cupy = _cupy()
    if isinstance(value, cupy.ndarray):
        return cupy.asnumpy(value)
    return value


def to_device(value: Any) -> Any:
    """把 host 端结果搬回设备（``numpy`` 数组/标量 -> ``cupy.ndarray``）。

    递归处理 ``tuple`` / ``list``（``numpy.linalg`` 会返回 namedtuple），
    非 NumPy 的 Python 值原样返回。
    """
    if isinstance(value, (numpy.ndarray, numpy.generic)):
        cupy = _cupy()
        return cupy.asarray(value)
    if isinstance(value, tuple):
        items = tuple(to_device(item) for item in value)
        maker = getattr(type(value), '_make', None)    # namedtuple 保结构
        if maker is not None:
            return maker(items)
        return items
    if isinstance(value, list):
        return [to_device(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# 派发
# ---------------------------------------------------------------------------
def run(numpy_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """跳过注册表，直接把 ``numpy_func`` 当作 host 实现跑一次。

    参数里的 ``cupy.ndarray`` 先 D2H，返回值里出现 ``numpy`` 数组/标量时再 H2D。
    """
    if not active():
        raise NotImplementedError(
            'CPU fallback 未启用（后端非 Ascend 或设置了 {}=1）'.format(
                DISABLE_ENV))
    np_args = [to_numpy(arg) for arg in args]
    np_kwargs = {key: to_numpy(val) for key, val in kwargs.items()}
    return to_device(numpy_func(*np_args, **np_kwargs))


def call(name: str, *args: Any, **kwargs: Any) -> Any:
    """按名字查 :data:`FALLBACKS` 并执行 host 计算，结果搬回设备。

    Args:
        name: 注册名，例如 ``'linalg.det'``。
        *args: 位置参数（``cupy.ndarray`` 会被搬到 host）。
        **kwargs: 关键字参数（同样会被搬运；``UPLO`` 之类直接透传给 NumPy）。
    """
    func = _check_active(name)
    return run(func, *args, **kwargs)
