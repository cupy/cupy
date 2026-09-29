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

def _host_in1d(ar1: Any, ar2: Any, assume_unique: bool = False,
               invert: bool = False) -> Any:
    """numpy.in1d 的 ravel 语义；用 numpy.isin 规避 numpy 2.x 的弃用告警。"""
    return numpy.isin(
        ar1, ar2, assume_unique=assume_unique, invert=invert).ravel()


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
    # cupy._logic.truth 的 set 系建立在 raw-CUDA ElementwiseKernel 上
    # （cupy_exists_kernel / cupy_exists_and_searchsorted_kernel /
    # setxorkernel），Ascend 没有对应 aclnn 算子，整个公开调用走 host
    # （接线见 truth._ascend_set_host_fallback；setdiff1d/isin 经 in1d
    # 自动覆盖，union1d 用 unique+concatenate，aclnn 已覆盖不需回退）。
    # in1d 的 ravel 语义用 numpy.isin 实现，规避 numpy 2.x 弃用告警。
    'logic.in1d': _host_in1d,
    'logic.intersect1d': numpy.intersect1d,
    'logic.setxor1d': numpy.setxor1d,
    # cupy._statistics.histogram 的非等宽 bin / weights / f64 等情形
    # aclnnHistc 表达不了（接线见 histogram._ascend_histogram）；
    # 等宽无权重且 dtype 受支持的主路径走 ascend_histc 设备直发。
    'statistics.histogram': numpy.histogram,
    # cupy._statistics.meanvar 的 var/std/nanmedian：aclnn 归约不收 complex
    # 输入，complex dtype 时整调用走 host（接线见 meanvar._ascend_complex_to_host）。
    'statistics.var': numpy.var,
    'statistics.std': numpy.std,
    'statistics.nanmedian': numpy.nanmedian,
    # cupy._statistics.histogram：bincount 的 aclnnBincount 不收 uint16/32/64
    # 输入与 complex 权重；histogramdd 的多段管线（searchsorted +
    # ravel_multi_index + bincount）没有单一 aclnn kernel，整调用走 host
    # （接线见 histogram._ascend_bincount / histogramdd）。
    'statistics.bincount': numpy.bincount,
    'statistics.histogramdd': numpy.histogramdd,
    # cupy._statistics.histogram.digitize：aclnnSearchSorted 只支持升序 bins，
    # numpy.digitize 还接受降序（接线见 histogram.digitize，2-scalar 探测方向）
    'statistics.digitize': numpy.digitize,
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
    'sum_with_dtype': numpy.sum,
    'prod_with_dtype': numpy.prod,
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
    'nansum': numpy.nansum,
    'nansum_with_dtype': numpy.nansum,
    'nanprod': numpy.nanprod,
    'nanprod_with_dtype': numpy.nanprod,
}


def f64_mode() -> str:
    """返回当前 float64 处理模式：``'off'`` / ``'float32'`` / ``'cpu'``。

    首次调用读取环境变量并缓存；用 :func:`reset_f64_mode` 清缓存。
    非法取值响亮报错（比静默忽略更安全）。
    default to float32, validated by pytest on 910B (only a few overflow failure)
    """
    global _f64_mode_cache
    if _f64_mode_cache is None:
        mode = os.getenv(F64_MODE_ENV, '').strip().lower()
        if mode == '':
            # 兼容旧开关：CUPY_ASCEND_ENABLE_FLOAT64_TO_FLOAT32=1 -> float32
            if os.getenv(LEGACY_F32_ENV, '0') == '1':
                mode = 'float32'
            elif active():
                mode = "float32" # default to float32 domoted
            else:
                mode = 'off'
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


def has_f64_io(*args: Any) -> bool:
    """``cpu`` 模式下 args 里是否有 float64/complex128 的设备数组。

    非 ``cpu`` 模式恒为 False（float32 降档由 acl_utils / _reduction 的
    既有 promote 层负责，与此互斥）。CScalar 等非 ndarray 跳过 —— 标量
    与 f32 数组混合时由 out 的 dtype 兜底判定。
    接受(in_args, out_args), (in_args, [ret])
    """
    if f64_mode() != 'cpu':
        return False
    cupy = _cupy()
    for seq in args:
        for x in seq:
            if isinstance(x, cupy.ndarray) and x.dtype.char in F64_DTYPES:
                return True
    return False


def _to_host(value: Any) -> Any:
    """设备数组 / CScalar / 其他 -> host 值。

    CScalar 经 ``CScalar.to_numpy_scalar``（cpdef，C 级解引用 ptr）得到
    NumPy 标量——ptr/size 是 cdef 属性，Python 层无法访问。
    非连续数据(non-contiguous): 先在设备侧物化成 C-连续副本再一次性 D2H
    （docs/ascend/Float64Workaround.md 修复 3 / P4-1）。
    """
    cupy = _cupy()
    if isinstance(value, cupy.ndarray):
        if value._c_contiguous:
            return cupy.asnumpy(value)
        if value.size == 0:
            return numpy.empty(value.shape, dtype=value.dtype)
        # 非连续视图的 nbytes = size*itemsize **小于**数据实际跨度
        # sum((shape[i]-1)*strides[i]) + itemsize。旧实现裸 memcpy nbytes
        # 字节再 as_strided 按跨度寻址：数据本身是错的，且在 nbytes 大小
        # 的 numpy 缓冲上越界读 -> 段错误。物化走 ascend_copy（实现即
        # aclnnCast，DOUBLE 原生支持；在 promote 豁免表
        # _UINT_PROMOTE_EXEMPT_OPS 内，且不经 ufunc 派发），不会递归回
        # cpu fallback 的拦截口。
        from cupy.backends.ascend.api.acl_utils import py_launch_general
        contig = cupy.empty(value.shape, dtype=value.dtype)
        py_launch_general('ascend_copy', (value,), (contig,), (), {})
        return cupy.asnumpy(contig)

    to_numpy_scalar = getattr(value, 'to_numpy_scalar', None)
    if to_numpy_scalar is not None:
        # CScalar: ptr/size 是 cdef 属性 Python 层访问不到，必须走
        # _scalar.pyx 的 cpdef 方法（C 级解引用）拿 NumPy 标量。
        return to_numpy_scalar()
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
        # ndarray.set() instead of `dev_out[...]= np_out` will use cupy.fill
        # may trigger `non-scalar numpy.ndarray` error
        dev_out.set(np_out)


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
    # do not use fill(value) as stated above,
    # also  make sure 0-D is returned
    ret.set(numpy.asarray(numpy.reshape(result, ret.shape)))
    return ret


# ---------------------------------------------------------------------------
# general 直发路径的 cpu 模式拦截（Float64Workaround.md §四-1 / 修复 4）
#
# 背景：general 直发调用点（_routines_indexing 的 take/index_select/
# scatter/nonzero/index_put、tiling 的 repeat、linalg、random 等）绕过
# _kernel.pyx/_reduction.pyx 的两个拦截口，cpu 模式下 f64 数组原样直达
# aclnn —— 对不收 DOUBLE 的 op 是段错误而不是报错。
#
# 拦截点：launch_general_func（checked 入口）。判据是 **任一** ins/outs
# 操作数为 float64/complex128 —— 不能只看第一个操作数：
#   * ascend_index_put_impl 的 ins = [indices, values]，ins[0] 是 int64
#     索引，f64 values 在 ins[1]（漏报 -> 段错误）；
#   * ascend_scatter_update 的 ins = [v, indices]，self 在 outs[0]。
# 误报方向是安全的：多拦（如 aclnnTake 原生支持 DOUBLE 也走 host）只是慢，
# 漏报就是崩溃。豁免表排除 aclnnCast 载体（ascend_cast/copy/positive）：
# 它们原生支持 DOUBLE，且是本模块 _to_host 与 acl_utils promote/cast-back
# 的内部实现，拦截会递归/自我破坏。
#
# 拦截后的动作：general ops 的 args 语义逐 op 各异（dim 列表、axis、
# accumulate 标志...），无法像 ufunc 那样 getattr(numpy, name) 通用回退，
# 只能 per-op 注册 host adapter；未注册的响亮 NotImplementedError（提示
# 改用 float32 降档）。
# ---------------------------------------------------------------------------

#: opname（ascend_ 前缀）-> host adapter ``(ins, outs, args, kwargs) -> None``。
#: adapter 把结果写进 outs（经 :func:`_copy_host_into`，对非连续 out 由
#: ascend_copy 的 general 通道 view write-back 负责回写）。
_GENERAL_HOST_FALLBACKS: Dict[str, Callable[..., Any]] = {}

#: aclnnCast 载体豁免（f64 合法；也是 cpu fallback 自身的内部通道）
_GENERAL_CPU_EXEMPT_OPS = frozenset((
    'ascend_cast',
    'ascend_copy', 
    'ascend_positive',
    'ascend_fill', # aclnnInplaceFillScalar support double/complex128 natively
))


def _normalize_opname(opname: str) -> str:
    """cupy_xxx -> ascend_xxx；ascend_xxx 原样返回。"""
    if opname.startswith('cupy_'):
        return 'ascend_' + opname[len('cupy_'):]
    return opname


def _copy_host_into(out: Any, np_result: Any) -> None:
    """host 结果 H2D 后经 ascend_copy 写进 out。

    ascend_copy（= aclnnCast）在 _GENERAL_CPU_EXEMPT_OPS 豁免表内，不会
    被本 gate 再次拦截；out 为偏移视图/非连续时由 general 通道的
    _wrap_materialize_outs -> _write_acl_out_to_view 负责物化回写。
    """
    cupy = _cupy()
    dev = cupy.asarray(
        numpy.ascontiguousarray(numpy.asarray(np_result), dtype=out.dtype))
    from cupy.backends.ascend.api.acl_utils import py_launch_general
    py_launch_general('ascend_copy', (dev,), (out,), (), {})


def _host_take(ins: Any, outs: Any, args: Any, kwargs: Any) -> None:
    """aclnnTake：self 视为 1-D flat，``out[i] = self[index[i]]``。

    调用点（_take 的 flat 分支）已保证 indices 非负且 in-bounds（wrap 过）。
    """
    np_a = numpy.asarray(_to_host(ins[0]))
    np_idx = numpy.asarray(_to_host(ins[1]))
    _copy_host_into(outs[0], np_a.ravel()[np_idx])


def _host_index_select(ins: Any, outs: Any, args: Any, kwargs: Any) -> None:
    """aclnnIndexSelect（torch.index_select 语义）：沿 dim 取行。"""
    dim = int(args[0]) if args else int(kwargs.get('dim', 0))
    np_self = numpy.asarray(_to_host(ins[0]))
    np_idx = numpy.asarray(_to_host(ins[1])).ravel()
    result = numpy.take(np_self, np_idx, axis=dim)
    _copy_host_into(outs[0], result.reshape(outs[0].shape))


def _host_nonzero(ins: Any, outs: Any, kwargs_ignored: Any = None,
                  *_args: Any, **_kwargs: Any) -> None:
    """aclnnNonzero：输出 (count, ndim) int64（调用方已同步 count 预分配）。

    注意 0-d 输入不会到达这里：count 非 0 时 ndim=0 的 dst.size==0 在
    调用点提前返回。
    """
    np_a = numpy.asarray(_to_host(ins[0]))
    idx = numpy.nonzero(np_a)
    _copy_host_into(outs[0], numpy.stack(idx, axis=1))


def _host_complex(ins: Any, outs: Any, args: Any, kwargs: Any) -> None:
    """aclnnComplex(re, im) -> complex（arange complex128 的 cpu 模式路径）。"""
    re = numpy.asarray(_to_host(ins[0]))
    im = numpy.asarray(_to_host(ins[1]))
    out = outs[0]
    result = numpy.empty(out.shape, dtype=out.dtype)
    result.real = re.reshape(result.shape)
    result.imag = im.reshape(result.shape)
    _copy_host_into(out, result)


# ---------------------------------------------------------------------------
# 稠密线代（B 类 structural：ascend_svd/qr/inverse/slogdet/trace/tril/triu）
#
# 这些 op 经 launch_general_func 直发，cpu 模式下 f64 操作数会被
# maybe_general_host 拦截 —— 此前没有 adapter，响亮 NotImplementedError。
# 但矩阵分解在 host 上「天然支持 float64」（numpy.linalg 原生双精度，
# LAPACK），注册 host adapter 后 cpu 模式即得到真 float64/complex128 结果，
# 与 linalg.eig/eigh 的既有 FALLBACKS 通道同一语义（D2H -> NumPy -> H2D，
# out dtype 由调用方按 cupy 语义预分配，_copy_host_into 保 dtype 回写）。
# ---------------------------------------------------------------------------

def _host_svd(ins: Any, outs: Any, args: Any, kwargs: Any) -> None:
    """aclnnSvd：ins=[a]，outs=[sigma]（svdvals）或 [sigma, u, v(=Vh)]，
    args=[full_matrices]（见 _routines_linalg._ascend_svd 的预分配）。"""
    np_a = numpy.asarray(_to_host(ins[0]))
    full_matrices = bool(args[0]) if args else True
    if len(outs) == 3:
        u, s, vh = numpy.linalg.svd(
            np_a, full_matrices=full_matrices, compute_uv=True)
        _copy_host_into(outs[0], s)
        _copy_host_into(outs[1], u)
        _copy_host_into(outs[2], vh)   # aclnn/cupy 的第三个 out 即 Vh
    else:
        s = numpy.linalg.svd(
            np_a, full_matrices=full_matrices, compute_uv=False)
        _copy_host_into(outs[0], s)


def _host_qr(ins: Any, outs: Any, args: Any, kwargs: Any) -> None:
    """aclnnQr：ins=[a]，outs=[q, r]，args=[some]（true == reduced）。"""
    np_a = numpy.asarray(_to_host(ins[0]))
    reduced = bool(args[0]) if args else True
    q, r = numpy.linalg.qr(
        np_a, mode='reduced' if reduced else 'complete')
    _copy_host_into(outs[0], q)
    _copy_host_into(outs[1], r)


def _host_inverse(ins: Any, outs: Any, args: Any, kwargs: Any) -> None:
    """aclnnInverse：ins=[a]，outs=[out]。"""
    np_a = numpy.asarray(_to_host(ins[0]))
    _copy_host_into(outs[0], numpy.linalg.inv(np_a))


def _host_slogdet(ins: Any, outs: Any, args: Any, kwargs: Any) -> None:
    """aclnnSlogdet：ins=[self]，outs=[sign, logdet]（见 _norms.slogdet）。"""
    np_a = numpy.asarray(_to_host(ins[0]))
    sign, logdet = numpy.linalg.slogdet(np_a)
    _copy_host_into(outs[0], sign)
    _copy_host_into(outs[1], logdet)


def _host_trace(ins: Any, outs: Any, args: Any, kwargs: Any) -> None:
    """aclnnTrace：ins=[a]，outs=[out]，args=[offset]（批量轴由
    numpy.trace 的 axis1=-2/axis2=-1 默认值覆盖）。"""
    np_a = numpy.asarray(_to_host(ins[0]))
    offset = int(args[0]) if args else 0
    _copy_host_into(outs[0], numpy.trace(np_a, offset=offset))


def _host_tril_triu(upper: bool) -> Callable[..., Any]:
    """aclnnTril/Triu：ins=[a]，outs=[out]，args=[k]。"""
    def _impl(ins: Any, outs: Any, args: Any, kwargs: Any) -> None:
        np_a = numpy.asarray(_to_host(ins[0]))
        k = int(args[0]) if args else 0
        fn = numpy.triu if upper else numpy.tril
        _copy_host_into(outs[0], fn(np_a, k))
    return _impl


_GENERAL_HOST_FALLBACKS.update({
    'ascend_take': _host_take,
    'ascend_index_select': _host_index_select,
    'ascend_nonzero': _host_nonzero,
    'ascend_complex': _host_complex,
    # 稠密线代：cpu 模式 f64 走 numpy.linalg（真双精度），f32 照常直达 aclnn
    'ascend_svd': _host_svd,
    'ascend_qr': _host_qr,
    'ascend_inverse': _host_inverse,
    'ascend_slogdet': _host_slogdet,
    'ascend_trace': _host_trace,
    'ascend_tril': _host_tril_triu(False),
    'ascend_triu': _host_tril_triu(True),
})


def maybe_general_host(opname: str, ins: Any, outs: Any,
                       args: Any, kwargs: Any) -> bool:
    """``launch_general_func`` 的 cpu 模式拦截 gate（供 acl_utils.pyx 调用）。

    Returns:
        True  — 已在 host 端执行完毕（cpu 模式 + f64 操作数 + 有注册的
                adapter），调用方直接返回；
        False — 无需拦截（非 cpu 模式 / 无 f64 操作数 / op 在豁免表），
                继续走 aclnn。

    Raises:
        NotImplementedError — cpu 模式 + f64 操作数 + 无 host adapter。
        相比让 float64 直达 aclnn 段错误，这里响亮失败并给出指引。
    """
    if f64_mode() != 'cpu':
        return False
    if not has_f64_io((ins, outs)):
        return False
    key = _normalize_opname(opname)
    if key in _GENERAL_CPU_EXEMPT_OPS:
        return False
    adapter = _GENERAL_HOST_FALLBACKS.get(key)
    if adapter is None:
        raise NotImplementedError(
            '{}: cpu 模式（{}=cpu）下 general 直发路径没有注册 host 实现'
            '（已注册: {}）。float64/complex128 操作数直达 aclnn 会段错误，'
            '故响亮报错；可在 cpu_fallback._GENERAL_HOST_FALLBACKS 添加 '
            'adapter，或改用 float32 降档模式'.format(
                opname, F64_MODE_ENV,
                ', '.join(sorted(_GENERAL_HOST_FALLBACKS))))
    adapter(ins, outs, args, kwargs)
    return True
