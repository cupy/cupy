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

import inspect
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


# ---------------------------------------------------------------------------
# float64 / complex128 CPU fallback（三态模式）
#
# 背景：910B 无 float64 硬件吞吐，且部分 aclnn 算子不收 DOUBLE/COMPLEX128。
# 两条路：
#   * float32 模式（旧机制）：CUPY_ASCEND_ENABLE_FLOAT64_TO_FLOAT32=1，在
#     acl_utils 派发层降档计算、结果 cast 回 f64 —— 精度损失，性能尚可；
#   * cpu 模式（本节）：CUPY_ASCEND_FLOAT64_MODE=cpu，在 `_core._ascend`
#     的两个派发口（_kernel.pyx ufunc、_reduction.pyx 归约）整 op 拦截，
#     D2H -> NumPy（真 float64 精度）-> H2D。
#
# 拦截发生在 acl_utils 的 promote 层之前，故 cpu 模式下 acl_utils 零改动。
# device 侧 f64 的创建/拷贝不需要 fallback：aclnnCast/aclnnCopy/aclnnFillScalar
# 均支持 DOUBLE/COMPLEX128，empty 是纯内存分配（见 CANN 9.0.1 头文件注释）。
# ---------------------------------------------------------------------------

#: 三态模式环境变量：off（默认，响亮报错）/ float32（降档）/ cpu（host 计算）
F64_MODE_ENV = 'CUPY_ASCEND_FLOAT64_MODE'

#: 旧开关（等价于 CUPY_ASCEND_FLOAT64_MODE=float32）
LEGACY_F32_ENV = 'CUPY_ASCEND_ENABLE_FLOAT64_TO_FLOAT32'

#: 参与 cpu fallback 判定的 dtype（float64、complex128 的 dtype char）
F64_DTYPES = frozenset('dG')

_f64_mode_cache: Optional[str] = None

#: 归约名（去 cupy_ 前缀）-> NumPy 实现。与 _kernel.pyx 的
#: `.reduce()` 名字映射（cupy_max -> array.max）同一约定。
_REDUCTION_HOST_MAP: Dict[str, Callable[..., Any]] = {
    'sum': numpy.sum,
    'prod': numpy.prod,
    'max': numpy.max,
    'min': numpy.min,
    'amax': numpy.amax,
    'amin': numpy.amin,
    'argmax': numpy.argmax,
    'argmin': numpy.argmin,
    'mean': numpy.mean,
    'var': numpy.var,
    'std': numpy.std,
    'nanmax': numpy.nanmax,
    'nanmin': numpy.nanmin,
    'nanmean': numpy.nanmean,
    'nanvar': numpy.nanvar,
    'nanstd': numpy.nanstd,
    'nanargmax': numpy.nanargmax,
    'nanargmin': numpy.nanargmin,
}


def f64_mode() -> str:
    """返回当前 float64 处理模式：``'off'`` / ``'float32'`` / ``'cpu'``。

    首次调用读取环境变量并缓存；用 :func:`reset_f64_mode` 清缓存。
    非法取值响亮报错（比静默忽略更安全）。
    """
    global _f64_mode_cache
    if _f64_mode_cache is None:
        mode = os.getenv(F64_MODE_ENV, '').strip().lower()
        if mode == '':
            # 兼容旧开关：CUPY_ASCEND_ENABLE_FLOAT64_TO_FLOAT32=1 -> float32
            mode = ('float32'
                    if os.getenv(LEGACY_F32_ENV, '0') == '1' else 'off')
        if mode not in ('off', 'float32', 'cpu'):
            raise ValueError(
                '{}={!r} 无效，可选 off / float32 / cpu'.format(
                    F64_MODE_ENV, mode))
        _f64_mode_cache = mode
    return _f64_mode_cache


def reset_f64_mode() -> None:
    """清掉 :func:`f64_mode` 的缓存（测试/运行时改环境变量后调用）。"""
    global _f64_mode_cache
    _f64_mode_cache = None


def has_f64_io(args: Any) -> bool:
    """``cpu`` 模式下 args 里是否有 float64/complex128 的设备数组。

    非 ``cpu`` 模式恒为 False（float32 降档由 acl_utils / _reduction 的
    既有 promote 层负责，与此互斥）。CScalar 等非 ndarray 跳过 —— 标量
    与 f32 数组混合时由 out 的 dtype 兜底判定。
    """
    if f64_mode() != 'cpu':
        return False
    cupy = _cupy()
    for x in args:
        if isinstance(x, cupy.ndarray) and x.dtype.char in F64_DTYPES:
            return True
    return False


def _to_host(value: Any) -> Any:
    """设备数组 / CScalar / 其他 -> host 值。

    CScalar 的物化方式与 acl_utils.pyx 的「标量物化成 0-d 数组」路径一致
    （ptr/size/get_numpy_type + memcpy），不过这里直接得到 numpy 标量。
    """
    cupy = _cupy()
    if isinstance(value, cupy.ndarray):
        return cupy.asnumpy(value)
    get_numpy_type = getattr(value, 'get_numpy_type', None)
    if get_numpy_type is not None:
        import ctypes
        buf = numpy.empty(value.size, dtype=numpy.uint8)
        ctypes.memmove(buf.ctypes.data, value.ptr, value.size)
        return numpy.frombuffer(buf, dtype=get_numpy_type())[0]
    return value


def run_elementwise_host(name: str, ins: Any, outs: Any,
                         kwargs: Dict[str, Any]) -> None:
    """在 host 端用 NumPy 执行一个 elementwise ufunc，结果写回设备 out。

    Args:
        name: cupy ufunc 名（``cupy_add`` 等），去前缀后映射到 numpy。
        ins: 输入（ndarray / CScalar / python 标量混合）。
        outs: 设备端 out 数组（shape/dtype 已由 cupy 解析好）。
        kwargs: 透传关键字（``where`` 设备数组会被搬到 host）。

    不支持的 ufunc（numpy 无同名函数）响亮报错，并提示可切 float32 模式。
    """
    fname = name[len('cupy_'):] if name.startswith('cupy_') else name
    np_ufunc = getattr(numpy, fname, None)
    if not callable(np_ufunc):
        raise NotImplementedError(
            '{}: cpu 模式下没有对应的 numpy.{} 实现；'
            '可改用 {}=float32 降档计算'.format(name, fname, F64_MODE_ENV))
    cupy = _cupy()
    np_ins = [_to_host(x) for x in ins]
    # out 先 D2H 成 host 缓冲（同 shape/dtype），算完整体写回 ——
    # 避免逐元素 H2D，且 out 与输入重叠（inplace）时语义与 NumPy 一致。
    np_outs = [_to_host(o) for o in outs]
    np_kwargs = {key: _to_host(val) for key, val in kwargs.items()}
    # numpy.where 等函数不支持 out=（与 _kernel.pyx all-scalar 路径的
    # inspect.signature 探测同一模式）：直接计算，结果写回 host 缓冲。
    try:
        supports_out = 'out' in inspect.signature(np_ufunc).parameters
    except (ValueError, TypeError):
        supports_out = True
    if supports_out:
        if len(np_outs) == 1:
            np_ufunc(*np_ins, out=np_outs[0], **np_kwargs)
        else:
            np_ufunc(*np_ins, out=tuple(np_outs), **np_kwargs)
    else:
        result = np_ufunc(*np_ins, **np_kwargs)
        np_outs = list(result) if isinstance(result, tuple) else [result]
    for dev_out, np_out in zip(outs, np_outs):
        dev_out[...] = np_out


def run_reduction_host(name: str, in_args: Any, ret: Any,
                       axis: Any, keepdims: bool, dtype: Any) -> Any:
    """在 host 端用 NumPy 执行一次归约，结果写回设备 ``ret``。

    Args:
        name: 归约内核名（``cupy_sum`` / ``cupy_nanvar`` 等）。
        in_args: 输入（当前实现只取第一个数组输入）。
        ret: 设备端输出（shape/dtype 已按 cupy 语义解析好）。
        axis/keepdims/dtype: 原样透传给 NumPy（axis 支持 None/int/tuple）。
    """
    fname = name[len('cupy_'):] if name.startswith('cupy_') else name
    func = _REDUCTION_HOST_MAP.get(fname)
    if func is None:
        raise NotImplementedError(
            '{}: cpu 模式下没有注册对应的 NumPy 归约实现（已注册: {}）；'
            '可改用 {}=float32 降档计算'.format(
                name, ', '.join(sorted(_REDUCTION_HOST_MAP)), F64_MODE_ENV))
    np_kwargs: Dict[str, Any] = {}
    if dtype is not None:
        np_kwargs['dtype'] = dtype
    result = func(_to_host(in_args[0]), axis=axis,
                  keepdims=bool(keepdims), **np_kwargs)
    # ret 的 shape 已由 cupy 语义解析好；reshape 兜底（keepdims 布局差异）
    ret[...] = numpy.reshape(result, ret.shape)
    return ret
