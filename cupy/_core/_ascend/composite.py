"""Ascend(NPU) 组合算子：用已注册的算子拼出 NPU 缺失的算子。

背景
----
CANN 没有这些算子的 aclnn 实现，但它们可以由**已有算子**组合得到（成本低，不必自研
AscendC 内核）。项目里的同类先例：

* C++ 侧 `acl_reduction_ops.h` 的 `aclop_NanMin` / `aclop_NanMax` ——
  `NanToNum(±inf)` + `Min/Max`；本次新增的 `aclop_NanProd` —— `NanToNum(nan=1)` + `Prod`；
  以及 `aclop_IsNan` = `not_equal(x, x)`、`aclop_Copysign` = abs+neg+ge+s_where。
* Python 侧 `cupy/linalg/_decomposition.py` 的 `is_ascend()` 分支 —— 组合已有算子实现
  `qr/svd/inv`；`cupy/_core/_ascend/_routines_sorting.pyx` 的 `partition → sort` 回退。

本模块集中放**纯 Python** 的组合实现，只在 :func:`active` 为真（Ascend 后端）时被公开
API 调用（`cupy.nanargmax` / `nanargmin` / `nanmean` / `nanprod` / `choose` / `angle(deg=True)`）。
模块顶层只 import `numpy`，`cupy` 一律函数内惰性导入 —— 因此在任何后端构建里
import 它都无副作用（不会触发 CANN 库加载）。

三条约定（都来自"本机无 NPU、只能静态验证"的约束）
--------------------------------------------------
1. **避免标量操作数**：`cupy.where(cond, 0.0, a)` 这类要经过 ufunc 的 scalar 通道，
   而 `acl_general_ops.h` 的 `aclop_Where` 明确写着 "a scalar operand is not
   supported by the Ascend backend yet" —— 改用 `full_like` 造同形状哨兵数组；
   比较/四则这类**已注册的 SCALAR_BINARY_OP**（`==`、`%`、`*`）可以直接用。
2. **避免 bool 上的归约与 dtype 转换**：`Sum` / `Cast` 对 bool 的支持未验证 ——
   计数走"1 的数组 + NaN 占位 -> nansum"这条全浮点路径。
3. **不新增算子注册**：组合只调用 `cupy.*` 公开 API（各自都有 aclnn 注册）。
   依赖清单见 :data:`REQUIRED_OPS`，`tests/ascend/test_composite_ops.py` 会拿它
   与运行时注册表 `py_list_acl_ufuncs()` / `py_list_custom_kernels()` 对账。

验证等级：本机无 NPU，只能到 L3（编译 + import + 注册）；**数值正确性待 910B**。
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy

__all__ = [
    'REQUIRED_OPS',
    'active',
    'angle_deg',
    'choose',
    'nanargmax',
    'nanargmin',
    'nanmean',
    'reset_cache',
]

#: 每个组合实现依赖的算子（`ascend_` 前缀之后的名字）。
#: 供 L3 静态校验：这些名字必须能在 Ascend 的两张注册表里查到，否则组合跑不起来。
REQUIRED_OPS: dict[str, tuple] = {
    'nanargmax': ('isnan', 'where', 'argmax'),
    'nanargmin': ('isnan', 'where', 'argmin'),
    'nanmean': ('isnan', 'where', 'fill', 'nansum', 'true_divide'),
    'nanprod': ('nan_to_num', 'prod'),
    'choose': ('where', 'equal', 'clip', 'remainder', 'min', 'max'),
    'angle_deg': ('angle', 'multiply'),
}

_ASCEND: Optional[bool] = None


# ---------------------------------------------------------------------------
# 后端判断 / 公开 API 取用
# ---------------------------------------------------------------------------
def active() -> bool:
    """当前后端是否为 Ascend（结果缓存；测试里可直接替换本函数）。"""
    global _ASCEND
    if _ASCEND is None:
        try:
            from cupy.backends.backend.api.runtime import is_ascend
            _ASCEND = bool(is_ascend())
        except Exception:      # pragma: no cover - 非 ascend 环境
            _ASCEND = False
    return _ASCEND


def reset_cache() -> None:
    """清掉 :func:`active` 的缓存（供测试使用）。"""
    global _ASCEND
    _ASCEND = None


def _cupy():
    import cupy
    return cupy


def _public(name: str, *fallback_modules: str) -> Any:
    """取公开 API；顶层没导出时回退到已知内部模块。

    本分支的 `cupy/__init__.py` 被裁剪过（`where` / `clip` / `argmax` … 顶层缺失），
    但算子本身是可用的 —— 例如 `where` 在 `cupy._sorting.search`。
    """
    value = getattr(_cupy(), name, None)
    if value is not None:
        return value
    import importlib
    for module_name in fallback_modules:
        module = importlib.import_module(module_name)
        value = getattr(module, name, None)
        if value is not None:
            return value
    raise AttributeError(f'cupy 中找不到 {name}（顶层与回退模块都没有）')


def _where():
    return _public('where', 'cupy._sorting.search')


def _clip():
    return _public('clip', 'cupy._math.misc')


# ---------------------------------------------------------------------------
# nan* 系列
# ---------------------------------------------------------------------------
def _fill_nan(a, value):
    """把 NaN 换成 ``value``，返回新数组。

    用 ``where(mask, full_like(a, value), a)`` 而不是 ``nan_to_num(a, nan=value)``：
    后者的 `nan` 参数要走 ufunc kwargs 通道（Ascend 上未验证），而 `full_like`
    走的是已注册的 `aclnnFillScalar`、`where` 走已注册的 `SWhere`。
    """
    cupy = _cupy()
    mask = cupy.isnan(a)
    sentinel = cupy.full_like(a, value)
    return _where()(mask, sentinel, a)


def nanargmax(a, axis=None, dtype=None, out=None, keepdims=False):
    """``nanargmax``：把 NaN 视为 -inf 后取 argmax。

    ``cupy_nanargmax`` 在 Ascend 无注册（`aclnnArgMax` 不支持"忽略 NaN"），
    组合实现与原 kernel 语义一致：全 NaN 切片返回 0，而 CUDA 后端返回越界值、
    NumPy 抛 ``ValueError``（`cupy.nanargmax` 的文档已声明该差异）。
    """
    filled = _fill_nan(a, -numpy.inf)
    return filled.argmax(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def nanargmin(a, axis=None, dtype=None, out=None, keepdims=False):
    """``nanargmin``：把 NaN 视为 +inf 后取 argmin（见 :func:`nanargmax`）。"""
    filled = _fill_nan(a, numpy.inf)
    return filled.argmin(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    """``nanmean`` = ``nansum(a) / 非 NaN 个数``。

    ``cupy_nanmean`` 的 kernel 把"累加和 + 计数"融合进一个 reduction struct；
    Ascend 上用两个已注册归约组合。计数用"1 的数组 + NaN 占位 -> nansum"得到：
    既不直接对 bool 归约（CANN 的 `Sum` 对 bool 支持未验证），也不依赖
    bool->float 的 `Cast`（CANN 的 `Cast` 对 bool 支持未验证）。
    全 NaN 切片 -> 0/0 = nan，与 NumPy 一致。
    """
    cupy = _cupy()
    work = a if dtype is None else a.astype(dtype)
    total = cupy.nansum(work, axis=axis, keepdims=keepdims)
    ones = cupy.full_like(work, 1.0)
    nan_value = cupy.full_like(work, numpy.nan)
    counted = _where()(cupy.isnan(work), nan_value, ones)
    count = cupy.nansum(counted, axis=axis, keepdims=keepdims)
    result = total / count
    if out is not None:
        out[...] = result
        return out
    return result


# ---------------------------------------------------------------------------
# choose
# ---------------------------------------------------------------------------
def choose(a, choices: Sequence[Any], out=None, mode: str = 'raise'):
    """``numpy.choose`` 的组合实现。

    ``cupy_choose`` / ``cupy_choose_clip`` 是 ElementwiseKernel（形参里有
    ``raw T choices`` 裸指针，按 ``choices`` 二维展平），aclnn 无法等价表达；
    这里逐 choice 用 ``where`` 选择：``n`` 个 choice 需要 ``n-1`` 次 ``where``，
    对 ``choose`` 这种低频 API 足够。

    Args:
        a: 索引数组（整型）。
        choices: 候选数组序列（数量 = n），需能广播到同一形状。
        out: 输出数组。
        mode: ``'raise'``（越界报错，默认）/ ``'wrap'``（取模）/ ``'clip'``（截断）。
    """
    cupy = _cupy()
    if isinstance(choices, (list, tuple)) is False:
        choices = list(choices)
    n = len(choices)
    if n == 0:
        raise ValueError('choices must be non-empty')

    a = cupy.asarray(a)
    choices = [cupy.asarray(choice) for choice in choices]
    shape = numpy.broadcast_shapes(a.shape, *[c.shape for c in choices])
    index = cupy.broadcast_to(a, shape)

    if mode == 'wrap':
        index = index % n
    elif mode == 'clip':
        index = _clip()(index, 0, n - 1)
    elif mode == 'raise':
        lo, hi = int(cupy.min(index)), int(cupy.max(index))
        if lo < 0 or hi >= n:
            raise ValueError('invalid entry in choice array')
    else:
        raise ValueError(f"mode must be one of 'raise', 'wrap', 'clip' (got {mode!r})")

    dtype = numpy.result_type(*choices)
    result = None
    for k, choice in enumerate(choices):
        candidate = cupy.broadcast_to(choice, shape).astype(dtype)
        if k == 0:
            result = candidate
        else:
            result = _where()(index == k, candidate, result)

    if out is not None:
        out[...] = result
        return out
    return result


# ---------------------------------------------------------------------------
# angle(deg=True)
# ---------------------------------------------------------------------------
def angle_deg(z):
    """``cupy.angle(z, deg=True)``：``angle(z) * 180 / pi``。

    ``cupy_angle_deg`` ufunc 无 aclnn 注册，但 ``cupy_angle`` 已由自定义 AscendC
    内核覆盖（`CUSTOM_UFUNCS['angle']`），再乘一个常数即可（常数走已注册的
    `SCALAR_BINARY_OP`，即 `ascend_multiply`）。

    常数必须收敛成 Python ``float``（强类型标量）：如果参与运算的是
    ``numpy.float64`` 之类的 0-d 强类型标量，dtype 提升会无视输入 dtype
    把整个结果强推成 float64——``angle(float32 输入)`` 本应返回 float32
    （实数路径见 acl_utils.pyx `_launch_custom_ufunc` 的实数分支，arctan2
    组合保持输入 dtype）。`numpy.pi` 虽是 Python float，但为防御后续改成
    `numpy` 标量表达式，这里显式 ``float(...)`` 收敛。
    """
    from cupy import _core
    return _core.angle(z) * float(180.0 / numpy.pi)
