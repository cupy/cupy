#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""numpy-ascend benchmark: CPU(numpy) vs XPU(cupy/ascend) speedup.

本文件的重构由下面这段提示词驱动, 原文保留在文件头:

    重构benchmark.py, 首先构建 reduction_op, binary_op, unary_op, 从op和dtype两个维度,
    测试math, matmul, cos, bitwise, bool, manipulation, sort等, 输出cpu 和xpu的加速比,
    vector 10M, matrix 4K, dtype重点float32全量, int64, float64加减乘除npu支持,也测试一下.

================================================================================
设计
================================================================================
1) 两个维度
   - op 维度: `TABLES` 按 `unary_op / binary_op / reduction_op / matmul /
     manipulation / sort` 分组, 组内再分 math / bitwise / bool;
     每个算子由 `OpSpec` 描述 (名字 / 参数个数 / 适用 dtype / 数据形状)。
   - dtype 维度: `OpSpec.dtypes` 显式声明该算子要跑的 dtype, 交叉相乘后逐例测试。
       * float32: 全量覆盖 (所有算子在 float32 上都跑一遍);
       * float64 / int64: 只挂在 add/subtract/multiply/true_divide 上 ——
         Ascend NPU 目前实测只支持这四则运算的 64-bit dtype (见 README.md §limitation);
       * int32: bitwise / 比较 / shift 等整型算子;
       * bool: logical / all / any 等布尔算子。
2) 数据规模: vector = 10M 元素 (`--vector-size`), matrix = 4K x 4K (`--matrix-size`)。
   matmul 用矩阵, 其余用向量; `--matrix-elementwise` 可让 elementwise/reduction
   额外在 4K 矩阵上再跑一遍。
3) 输出: 每例一行 -> CPU 平均耗时 / XPU 平均耗时 / 加速比 (cpu/xpu) / 数值一致性。
   末尾给出汇总、最高/最低加速比、以及未通过清单。
4) 健壮性: 每例独立 try/except。Ascend 上未移植的算子记 `UNSUPPORTED`,
   其它异常记 `FAIL`, 都不影响后续算子继续跑。
5) 计时: CPU 用 `time.perf_counter`; XPU 每轮 `synchronize` 后再计时 (测设备时间)。
   `--async` 改为「连续提交 N 次后同步一次」测吞吐 (提交流水)。
6) 测试数据: float 为 [0,1) 均匀分布 (定义域外的算子如 arccosh/arctanh 产生 NaN,
   用 equal_nan 比对); int32 取 0..7 (避免 shift/gcd/lcm 溢出), int64 取 1..99
   (避免除零); bool 为随机 0/1。数据按 (dtype, 形状) 缓存复用。
7) 算子解析: numpy 用顶层 API; cupy 先查顶层, 再按 `_CUPY_SUBMODULES` 回退到子模块
   —— 本分支 `cupy/__init__.py` 裁剪过 (如 `cupy.absolute` 只在 `cupy._math.misc`),
   回退生效时 note 列会标出 `顶层未导出 (via cupy.xxx)`, 便于后续补齐导出。

================================================================================
用法
================================================================================
    python benchmark.py                       # 全量跑 (需要 NPU)
    python benchmark.py --list                # 只打印 op x dtype 矩阵 (无需 NPU)
    python benchmark.py --category unary_op    # 只跑 unary_op/* 组
    python benchmark.py --dtype float32
    python benchmark.py --repeat 5 --vector-size 1000000 --matrix-size 1024
    python benchmark.py --matrix-elementwise   # elementwise 也在 4K 矩阵上跑
    python benchmark.py --csv benchmark.csv
    python benchmark.py --strict               # 有 FAIL/MISMATCH 时返回非 0

退出码: 0 = 跑完; 1 = --strict 下出现 FAIL/MISMATCH;
        2 = cupy 导入失败; 3 = 无可用 NPU 设备 (此时请用 --list 查看算子矩阵)。

注意: 本机 (无 NPU) 只能跑到 L3 (编译 + import + 算子注册), 真实数值/加速比需在
910B 上运行本脚本; 脚本不会声称数值正确性, 只报告 allclose 的结果。
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# XPU (cupy) import —— 允许在没装 CANN 的机器上执行 --list
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    from cupy import xpu
except Exception as _exc:  # pragma: no cover - 取决于机器环境
    cp = None
    xpu = None
    _CUPY_IMPORT_ERROR: Optional[BaseException] = _exc
else:
    _CUPY_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------
VECTOR_SIZE = 10_000_000      # 10M 元素
MATRIX_SIZE = 4096            # 4K x 4K
VECTOR_REPEAT = 10            # 向量/归约重复次数
MATRIX_REPEAT = 3             # 矩阵重复次数 (CPU 端 4K matmul 很慢)
SORT_REPEAT = 2               # 排序类重复次数 (10M CPU 排序很慢)

F32 = ("float32",)
F64_I64 = ("float32", "float64", "int64")   # NPU: 64-bit 仅四则运算
I32 = ("int32",)
F32_I32 = ("float32", "int32")              # 比较类: 浮点 + 整型
BOOL = ("bool",)

#: dtype 标签 -> numpy dtype
DTYPES: dict[str, Any] = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
}

#: allclose 容差 (rtol, atol), 按 dtype
TOLERANCE: dict[str, tuple[float, float]] = {
    "float32": (1e-3, 1e-4),
    "float64": (1e-9, 1e-12),
}

#: cupy/__init__.py 被裁剪后, 不少算子只有底层模块里有; 按这些模块回退查找
_CUPY_SUBMODULES = (
    "cupy._math.misc", "cupy._math.arithmetic", "cupy._math.explog",
    "cupy._math.trigonometric", "cupy._math.hyperbolic", "cupy._math.rounding",
    "cupy._math.special", "cupy._math.floating", "cupy._math.rational",
    "cupy._math.sumprod", "cupy._math.window", "cupy._math.ufunc",
    "cupy._logic.comparison", "cupy._logic.content", "cupy._logic.ops",
    "cupy._logic.truth", "cupy._logic.type_testing",
    "cupy._binary.elementwise", "cupy._binary.packing",
    "cupy._statistics.order", "cupy._statistics.meanvar",
    "cupy._statistics.correlation",
    "cupy._indexing.indexing", "cupy._indexing.generate",
    "cupy._creation.matrix", "cupy._creation.basic", "cupy._creation.ranges",
    "cupy._manipulation.basic", "cupy._manipulation.dims",
    "cupy._manipulation.join", "cupy._manipulation.kind",
    "cupy._manipulation.rearrange", "cupy._manipulation.shape",
    "cupy._manipulation.split", "cupy._manipulation.tiling",
    "cupy._manipulation.transpose",
    "cupy._sorting.sort", "cupy._sorting.search", "cupy._sorting.count",
)


# ---------------------------------------------------------------------------
# 算子解析: numpy 用顶层, cupy 走「顶层 + 子模块回退」
# ---------------------------------------------------------------------------
_CUPY_MODULE_CACHE: Optional[list] = None


def _cupy_modules() -> list:
    """惰性导入候选 cupy 子模块 (失败静默跳过)。"""
    global _CUPY_MODULE_CACHE
    if _CUPY_MODULE_CACHE is None:
        mods = [cp]
        for name in _CUPY_SUBMODULES:
            try:
                mods.append(importlib.import_module(name))
            except Exception:
                pass
        _CUPY_MODULE_CACHE = mods
    return _CUPY_MODULE_CACHE


def resolve_numpy(name: str) -> tuple[Optional[Any], Optional[str]]:
    value = getattr(np, name, None)
    return (value, "numpy") if value is not None else (None, None)


def resolve_cupy(name: str) -> tuple[Optional[Any], Optional[str]]:
    if cp is None:
        return None, None
    for mod in _cupy_modules():
        value = getattr(mod, name, None)
        if value is None:
            continue
        # 顶层接受任意属性 (dtype 等), 子模块只接受可调用的算子
        if mod is cp or callable(value):
            return value, mod.__name__
    return None, None


class XpNamespace:
    """把 numpy / cupy 包成统一的命名空间, 供 `fn(xp, a, b, c)` 使用。"""

    def __init__(self, name: str, resolver: Callable[[str], tuple]):
        self._name = name
        self._resolver = resolver

    @property
    def name(self) -> str:
        return self._name

    def resolve(self, attr: str) -> tuple[Any, str]:
        value, origin = self._resolver(attr)
        if value is None:
            raise AttributeError(f"{self._name} 无此算子/属性: {attr}")
        return value, origin

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("_"):
            raise AttributeError(attr)
        return self.resolve(attr)[0]


_NP = XpNamespace("numpy", resolve_numpy)
_CP = XpNamespace("cupy", resolve_cupy)


# ---------------------------------------------------------------------------
# 算子表: op x dtype 两个维度
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpSpec:
    """一个待测算子。

    Args:
        name: 公共 API 名字 (numpy 与 cupy 同名)。
        kind: 'unary' (1 个操作数) | 'binary' (2 个操作数) | 'reduce' (归约);
             决定默认调用方式 `f(a)` / `f(a, b)` 以及矩阵扩展行为。
        dtypes: 该算子要测的 dtype 标签。
        where: 'vector' (10M) 或 'matrix' (4K x 4K)。
        repeat: 覆盖默认重复次数 (None 用 CLI 默认值)。
        fn: 自定义调用 `fn(xp_namespace, a, b, c)`, 用于带额外参数 / 复合算子。
        note: 备注 (会打印到 note 列)。
    """

    name: str
    kind: str
    dtypes: tuple = F32
    where: str = "vector"
    repeat: Optional[int] = None
    fn: Optional[Callable] = None
    note: str = ""


def _specs(names: Sequence[str], kind: str, **kw) -> list:
    return [OpSpec(name, kind, **kw) for name in names]


def _words(text: str) -> list:
    return text.split()


#: 归约类默认全量归约 (无 axis), fn 里可覆盖
TABLES: dict[str, list[OpSpec]] = {
    # ---------------- unary_op ----------------
    "unary_op/math": _specs(_words("""
        cos sin tan cosh sinh tanh
        arccos arcsin arctan arccosh arcsinh arctanh
        exp exp2 expm1 log log2 log10 log1p
        sqrt cbrt square reciprocal absolute negative positive sign
        floor ceil trunc rint round nan_to_num
        deg2rad rad2deg sinc
    """), "unary"),
    "unary_op/predicate": _specs(
        _words("isnan isinf isfinite signbit"), "unary", note="输出 bool"),
    "unary_op/bitwise": _specs(
        _words("invert bitwise_not"), "unary", dtypes=I32),
    "unary_op/bool": _specs(
        _words("logical_not"), "unary", dtypes=BOOL),

    # ---------------- binary_op ----------------
    "binary_op/math": [
        # int64 / float64: NPU 目前只支持四则运算
        *_specs(_words("add subtract multiply true_divide"), "binary",
                dtypes=F64_I64, note="含 float64/int64 (NPU 四则运算)"),
        *_specs(_words("""
            floor_divide power float_power fmod remainder
            maximum minimum arctan2 hypot copysign fmax fmin
            logaddexp logaddexp2 heaviside
        """), "binary"),
    ],
    "binary_op/bitwise": _specs(
        _words("bitwise_and bitwise_or bitwise_xor left_shift right_shift gcd lcm"),
        "binary", dtypes=I32, note="整型算子"),
    "binary_op/bool": [
        *_specs(_words("greater less greater_equal less_equal equal not_equal"),
                "binary", dtypes=F32_I32, note="输出 bool"),
        *_specs(_words("logical_and logical_or logical_xor"), "binary",
                dtypes=BOOL, note="输入 bool"),
        *_specs(_words("isclose"), "binary", dtypes=F32),
    ],

    # ---------------- reduction_op ----------------
    "reduction_op/math": _specs(_words("""
        sum prod mean max min argmax argmin
        cumsum cumprod ptp std var
        nansum nanprod nanmax nanmin nancumsum nancumprod
    """), "reduce"),
    "reduction_op/bool": _specs(
        _words("all any"), "reduce", dtypes=BOOL, note="输入 bool"),

    # ---------------- matmul (matrix 4K) ----------------
    "matmul": _specs(
        _words("matmul dot"), "binary", where="matrix", repeat=MATRIX_REPEAT),

    # ---------------- manipulation ----------------
    "manipulation": [
        OpSpec("concatenate", "binary",
               fn=lambda xp, a, b, c: xp.concatenate([a, b]), note="[a,b]"),
        OpSpec("stack", "binary",
               fn=lambda xp, a, b, c: xp.stack([a, b]), note="[a,b]"),
        OpSpec("flip", "unary", fn=lambda xp, a, b, c: xp.flip(a)),
        OpSpec("roll", "unary", fn=lambda xp, a, b, c: xp.roll(a, 5), note="shift=5"),
        OpSpec("take", "unary", fn=lambda xp, a, b, c: xp.take(a, c),
               note="1/10 索引"),
        OpSpec("repeat", "unary", fn=lambda xp, a, b, c: xp.repeat(a, 2),
               note="repeats=2"),
        OpSpec("tile", "unary", fn=lambda xp, a, b, c: xp.tile(a, 2), note="reps=2"),
        OpSpec("copy", "unary", fn=lambda xp, a, b, c: xp.copy(a)),
        OpSpec("clip", "unary", fn=lambda xp, a, b, c: xp.clip(a, 0.2, 0.8)),
        OpSpec("where", "binary",
               fn=lambda xp, a, b, c: xp.where(a > 0.5, a, b),
               note="含 a>0.5 比较"),
        OpSpec("astype", "unary", fn=lambda xp, a, b, c: a.astype(xp.float64),
               dtypes=F32, note="f32->f64 cast"),
        OpSpec("tril", "unary", where="matrix", repeat=MATRIX_REPEAT),
        OpSpec("triu", "unary", where="matrix", repeat=MATRIX_REPEAT),
        OpSpec("transpose", "unary", where="matrix", repeat=MATRIX_REPEAT),
    ],

    # ---------------- sort ----------------
    "sort": [
        *_specs(_words("sort argsort"), "unary", repeat=SORT_REPEAT),
        OpSpec("partition", "unary", repeat=SORT_REPEAT,
               fn=lambda xp, a, b, c: xp.partition(a, a.size // 2),
               note="kth=size/2"),
        OpSpec("argpartition", "unary", repeat=SORT_REPEAT,
               fn=lambda xp, a, b, c: xp.argpartition(a, a.size // 2),
               note="kth=size/2"),
    ],
}


# ---------------------------------------------------------------------------
# 结果与数据
# ---------------------------------------------------------------------------
@dataclass
class Result:
    category: str
    op: str
    dtype: str
    where: str
    size: str
    cpu_mean: float = 0.0
    cpu_min: float = 0.0
    xpu_mean: float = 0.0
    xpu_min: float = 0.0
    speedup: float = 0.0
    status: str = "OK"
    note: str = ""


class DataCache:
    """按 (dtype, 形状) 缓存 host / device 测试数据。"""

    def __init__(self, vector_size: int, matrix_size: int):
        self.vector_size = vector_size
        self.matrix_size = matrix_size
        self._host: dict = {}
        self._device: dict = {}

    def _shape(self, where: str) -> tuple:
        if where == "matrix":
            return (self.matrix_size, self.matrix_size)
        return (self.vector_size,)

    def host(self, dtype_label: str, where: str) -> tuple:
        key = (dtype_label, where)
        if key not in self._host:
            shape = self._shape(where)
            dt = DTYPES[dtype_label]
            kind = np.dtype(dt).kind
            if kind == "b":
                a = np.random.rand(*shape) < 0.5
                b = np.random.rand(*shape) < 0.5
            elif kind in "iu":
                # 小整数: 避免 shift/gcd/lcm 溢出; 非 0: 避免整型除零
                lo, hi = (0, 8) if dt is np.int32 else (1, 100)
                a = np.random.randint(lo, hi, shape).astype(dt)
                b = np.random.randint(lo, hi, shape).astype(dt)
            else:
                a = np.random.rand(*shape).astype(dt)
                b = np.random.rand(*shape).astype(dt)
            idx = np.arange(0, a.size, 10, dtype=np.int64)
            self._host[key] = (a.astype(dt, copy=False),
                               b.astype(dt, copy=False), idx)
        return self._host[key]

    def device(self, dtype_label: str, where: str) -> tuple:
        key = (dtype_label, where)
        if key not in self._device:
            host = self.host(dtype_label, where)
            self._device[key] = tuple(cp.asarray(x) for x in host)
        return self._device[key]


# ---------------------------------------------------------------------------
# 计时 / 调用 / 校验
# ---------------------------------------------------------------------------
def _sync() -> None:
    """等待 NPU 上所有已提交任务完成。"""
    if xpu is not None:
        xpu.Stream.null.synchronize()


def _bind(spec: OpSpec, xp: XpNamespace, a, b, c) -> tuple[Callable, str]:
    """把 OpSpec 绑定到具体数据, 返回 (无参调用, 算子来源模块)。"""
    if spec.fn is not None:
        return (lambda: spec.fn(xp, a, b, c)), xp.name
    func, origin = xp.resolve(spec.name)
    if spec.kind in ("unary", "reduce"):
        return (lambda: func(a)), origin
    return (lambda: func(a, b)), origin


def _measure(call: Callable, repeat: int, sync: Optional[Callable] = None,
             async_mode: bool = False) -> tuple[float, float, Any]:
    """返回 (最短耗时, 平均耗时, 最后一次输出), 单位秒。"""
    out = call()                      # warmup: JIT / workspace / cache
    if sync is not None:
        sync()
    if async_mode and sync is not None:
        start = time.perf_counter()
        for _ in range(repeat):
            out = call()
        sync()
        mean = (time.perf_counter() - start) / repeat
        return mean, mean, out
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        out = call()
        if sync is not None:
            sync()
        times.append(time.perf_counter() - start)
    return min(times), sum(times) / len(times), out


def _to_numpy(value) -> np.ndarray:
    """把 XPU 结果搬回 host, 用于一致性校验。"""
    if cp is not None and hasattr(cp, "asnumpy"):
        try:
            return np.asarray(cp.asnumpy(value))
        except Exception:
            pass
    return np.asarray(value)


def _check(cpu_out, xpu_out, dtype_label: str) -> tuple[bool, str]:
    """比对 CPU / XPU 结果。"""
    lhs = np.asarray(cpu_out)
    rhs = _to_numpy(xpu_out)
    if lhs.shape != rhs.shape:
        return False, f"shape {lhs.shape} != {rhs.shape}"
    if lhs.dtype.kind in "biu" or rhs.dtype.kind in "biu":
        return bool(np.array_equal(lhs, rhs)), "值不一致"
    rtol, atol = TOLERANCE.get(dtype_label, (1e-3, 1e-4))
    ok = bool(np.allclose(lhs, rhs, rtol=rtol, atol=atol, equal_nan=True))
    return ok, f"allclose(rtol={rtol:g}, atol={atol:g}) 失败"


def _brief(exc: BaseException, limit: int = 110) -> str:
    """一行摘要; 过长时保留首尾 (CANN 的关键错误码在尾部)。"""
    text = " ".join(str(exc).split())
    name = type(exc).__name__
    if not text:
        return name
    if len(text) <= limit:
        return f"{name}: {text}"
    half = (limit - 5) // 2
    return f"{name}: {text[:half]} ... {text[-half:]}"


_UNSUPPORTED_HINTS = (
    "not registered", "no ascend", "notimplemented", "not implemented",
    "unsupported", "el0003", "invalid_argument", "unimplemented",
)


def _is_unsupported(exc: BaseException) -> bool:
    if isinstance(exc, NotImplementedError):
        return True
    text = str(exc).lower()
    return any(hint in text for hint in _UNSUPPORTED_HINTS)


def _count_label(n: int) -> str:
    if n >= 1024 and n % 1024 == 0:            # 4096 -> 4K
        if n >= 1024 * 1024:
            return f"{n // (1024 * 1024)}M"
        return f"{n // 1024}K"
    if n >= 1_000_000:
        return f"{n / 1e6:g}M"
    if n >= 1_000:
        return f"{n / 1e3:g}K"
    return str(n)


def _size_label(where: str, args) -> str:
    if where == "matrix":
        label = _count_label(args.matrix_size)
        return f"{label}x{label}"
    return _count_label(args.vector_size)


def _wheres(spec: OpSpec, args) -> tuple:
    if spec.where == "matrix":
        return ("matrix",)
    if args.matrix_elementwise and spec.kind in ("unary", "binary", "reduce"):
        return ("vector", "matrix")
    return ("vector",)


# ---------------------------------------------------------------------------
# 单例执行
# ---------------------------------------------------------------------------
def run_case(spec: OpSpec, dtype_label: str, where: str,
             cache: DataCache, args) -> Result:
    repeat = spec.repeat or (
        args.matrix_repeat if where == "matrix" else args.repeat)
    row = Result(category="", op=spec.name, dtype=dtype_label,
                 where=where, size=_size_label(where, args))
    if spec.note:
        row.note = spec.note

    # --- host 数据
    try:
        host = cache.host(dtype_label, where)
    except Exception as exc:
        row.status, row.note = "SKIP", f"数据生成失败: {_brief(exc)}"
        return row

    # --- CPU (numpy)
    try:
        np_call, _ = _bind(spec, _NP, *host)
    except AttributeError as exc:
        row.status, row.note = "SKIP", f"numpy: {_brief(exc)}"
        return row
    try:
        with np.errstate(all="ignore"):     # 定义域外输入 (如 arccosh(0..1)) 会产生 NaN
            cpu_min, cpu_mean, cpu_out = _measure(np_call, repeat)
    except Exception as exc:
        row.status, row.note = "SKIP", f"numpy 执行失败: {_brief(exc)}"
        return row

    # --- XPU (cupy)
    try:
        device = cache.device(dtype_label, where)
    except Exception as exc:
        row.status, row.note = "SKIP", f"device: {_brief(exc)}"
        return row
    try:
        cp_call, origin = _bind(spec, _CP, *device)
    except AttributeError as exc:
        row.status, row.note = "SKIP", f"cupy: {_brief(exc)}"
        return row
    try:
        xpu_min, xpu_mean, xpu_out = _measure(
            cp_call, repeat, sync=_sync, async_mode=args.async_mode)
    except Exception as exc:
        row.status = "UNSUPPORTED" if _is_unsupported(exc) else "FAIL"
        row.note = _brief(exc)
        return row

    row.cpu_mean, row.cpu_min = cpu_mean, cpu_min
    row.xpu_mean, row.xpu_min = xpu_mean, xpu_min
    row.speedup = cpu_mean / xpu_mean if xpu_mean > 0 else float("inf")
    if origin not in ("cupy",):
        note = f"顶层未导出 (via {origin})"
        row.note = f"{row.note}; {note}" if row.note else note

    if not args.no_check:
        ok, why = _check(cpu_out, xpu_out, dtype_label)
        if not ok:
            row.status = "MISMATCH"
            row.note = f"{row.note}; {why}" if row.note else why
    return row


# ---------------------------------------------------------------------------
# 打印
# ---------------------------------------------------------------------------
_WIDTH = (18, 15, 8, 8, 10, 10, 9, 12)


def _print_result_row(row: Result) -> None:
    speedup = ("inf" if row.speedup == float("inf") else f"{row.speedup:.1f}x"
               ) if row.xpu_mean > 0 else "-"
    cells = (
        f"{row.category:<{_WIDTH[0]}}",
        f"{row.op:<{_WIDTH[1]}}",
        f"{row.dtype:<{_WIDTH[2]}}",
        f"{row.size:<{_WIDTH[3]}}",
        f"{row.cpu_mean * 1e3:>{_WIDTH[4]}.4f}",
        f"{row.xpu_mean * 1e3:>{_WIDTH[5]}.4f}",
        f"{speedup:>{_WIDTH[6]}}",
        f"{row.status:<{_WIDTH[7]}}",
        row.note,
    )
    print(" ".join(c for c in cells).rstrip())


def _print_header(args) -> None:
    print("=" * 108)
    cupy_version = cp.__version__ if cp is not None else "N/A (import 失败)"
    print(f"numpy-ascend benchmark | numpy {np.__version__} | cupy {cupy_version}")
    print(f"vector={_count_label(args.vector_size)}  matrix="
          f"{_count_label(args.matrix_size)}x{_count_label(args.matrix_size)}  "
          f"repeat(vector/matrix)={args.repeat}/{args.matrix_repeat}  "
          f"mode={'async(吞吐)' if args.async_mode else 'sync(设备时间)'}"
          f"  check={'off' if args.no_check else 'on'}")
    print(f"backend: {_backend_label()}")
    print("=" * 108)
    print(f"{'category':<{_WIDTH[0]}} {'op':<{_WIDTH[1]}} {'dtype':<{_WIDTH[2]}} "
          f"{'size':<{_WIDTH[3]}} {'cpu_ms':>{_WIDTH[4]}} {'xpu_ms':>{_WIDTH[5]}} "
          f"{'speedup':>{_WIDTH[6]}} {'status':<{_WIDTH[7]}} note")
    print("-" * 108)


def _backend_label() -> str:
    if cp is None:
        return f"cupy 未导入: {_brief(_CUPY_IMPORT_ERROR)}"
    parts = []
    try:
        from cupy.backends.backend.api import runtime
        parts.append("ascend" if runtime.is_ascend() else "cuda/other")
    except Exception:
        parts.append("unknown")
    try:
        from cupy.backends.ascend import _detect_installed_version
        version = _detect_installed_version()
        if version is not None:
            parts.append(f"CANN {version // 100}.{version // 10 % 10}.{version % 10}")
    except Exception:
        pass
    try:
        n = len(__import__("cupy.backends.ascend.api.acl_utils",
                           fromlist=["x"]).py_list_acl_ufuncs())
        parts.append(f"registered ops={n}")
    except Exception:
        pass
    return " | ".join(parts)


def _print_list(args) -> None:
    """打印 op x dtype 矩阵 (不需要 NPU)。"""
    print(f"{'category':<20} {'op':<15} {'kind':<7} {'where':<7} "
          f"{'repeat':<7} {'dtypes':<28} resolve(cupy) / note")
    print("-" * 108)
    for category, specs in TABLES.items():
        for spec in specs:
            if args.category and args.category not in category:
                continue
            origin = "-"
            if cp is not None and spec.fn is None:
                _, origin = resolve_cupy(spec.name)
                origin = origin or "缺失"
            repeat = spec.repeat or "-"
            note = spec.note or ("composed" if spec.fn else "")
            print(f"{category:<20} {spec.name:<15} {spec.kind:<7} "
                  f"{spec.where:<7} {str(repeat):<7} "
                  f"{','.join(spec.dtypes):<28} {origin} {note}")
    total = sum(len(v) for v in TABLES.values())
    print("-" * 108)
    print(f"共 {len(TABLES)} 组 / {total} 个算子条目; "
          "dtype 维度由每条的 dtypes 决定 (float32 全量, float64/int64 仅四则运算)")


def _print_summary(results: list[Result], args) -> int:
    print("-" * 108)
    counts = Counter(r.status for r in results)
    summary = "  ".join(f"{k}={counts[k]}" for k in sorted(counts))
    print(f"总计 {len(results)} 例 | {summary}")
    ok_rows = [r for r in results if r.status == "OK"]
    if ok_rows:
        best = max(ok_rows, key=lambda r: r.speedup)
        worst = min(ok_rows, key=lambda r: r.speedup)
        print(f"最高加速: {best.op} {best.dtype} {best.size} -> "
              f"{best.speedup:.1f}x (cpu {best.cpu_mean * 1e3:.4f} ms / "
              f"xpu {best.xpu_mean * 1e3:.4f} ms)")
        print(f"最低加速: {worst.op} {worst.dtype} {worst.size} -> "
              f"{worst.speedup:.1f}x")
    bad = [r for r in results if r.status != "OK"]
    if bad:
        print("\n未通过清单 (status / op / dtype / size / 原因):")
        for r in bad:
            print(f"  {r.status:<12} {r.category:<18} {r.op:<15} {r.dtype:<8} "
                  f"{r.size:<8} {r.note}")
    if not args.no_check and ok_rows:
        print("\n注: OK 仅表示 allclose 通过; 本脚本不验证算法级数值正确性。")
    if args.strict and any(r.status in ("FAIL", "MISMATCH") for r in results):
        return 1
    return 0


def _write_csv(path: str, results: list[Result]) -> None:
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(("category", "op", "dtype", "where", "size",
                         "cpu_mean_ms", "cpu_min_ms", "xpu_mean_ms",
                         "xpu_min_ms", "speedup", "status", "note"))
        for r in results:
            writer.writerow((
                r.category, r.op, r.dtype, r.where, r.size,
                f"{r.cpu_mean * 1e3:.6f}", f"{r.cpu_min * 1e3:.6f}",
                f"{r.xpu_mean * 1e3:.6f}", f"{r.xpu_min * 1e3:.6f}",
                "inf" if r.speedup == float("inf") else f"{r.speedup:.4f}",
                r.status, r.note))
    print(f"\n结果已写入 {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="numpy(CPU) vs cupy/ascend(XPU) 加速比基准测试",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--vector-size", type=int, default=VECTOR_SIZE,
                        help="向量元素个数")
    parser.add_argument("--matrix-size", type=int, default=MATRIX_SIZE,
                        help="矩阵阶数 N (N x N)")
    parser.add_argument("--repeat", type=int, default=VECTOR_REPEAT,
                        help="向量/归约类重复次数")
    parser.add_argument("--matrix-repeat", type=int, default=MATRIX_REPEAT,
                        help="矩阵类重复次数")
    parser.add_argument("--category", default="",
                        help="只跑名字包含该字符串的分组, 如 unary_op / matmul")
    parser.add_argument("--dtype", default="",
                        help="只跑指定 dtype, 逗号分隔, 如 float32,int64")
    parser.add_argument("--list", dest="list_ops", action="store_true",
                        help="只打印算子 x dtype 矩阵, 不做任何设备运算")
    parser.add_argument("--matrix-elementwise", action="store_true",
                        help="elementwise/归约类额外在 4K 矩阵上跑一遍")
    parser.add_argument("--async", dest="async_mode", action="store_true",
                        help="连续提交 repeat 次后同步一次, 测吞吐")
    parser.add_argument("--no-check", action="store_true",
                        help="不做 CPU/XPU 数值一致性校验")
    parser.add_argument("--csv", default="", help="把结果写入 CSV 文件")
    parser.add_argument("--strict", action="store_true",
                        help="出现 FAIL/MISMATCH 时返回退出码 1")
    return parser.parse_args(argv)


def _selected_cases(args) -> list[tuple[str, OpSpec, str, str]]:
    """展开成 (category, spec, dtype, where) 用例列表。"""
    wanted_dtypes = {d.strip() for d in args.dtype.split(",") if d.strip()}
    cases = []
    for category, specs in TABLES.items():
        if args.category and args.category not in category:
            continue
        for spec in specs:
            for dtype_label in spec.dtypes:
                if wanted_dtypes and dtype_label not in wanted_dtypes:
                    continue
                for where in _wheres(spec, args):
                    cases.append((category, spec, dtype_label, where))
    return cases


def device_probe() -> tuple[bool, str]:
    """本机是否可真正执行 NPU 运算。"""
    if cp is None:
        return False, f"cupy 未导入: {_brief(_CUPY_IMPORT_ERROR)}"
    try:
        probe = cp.zeros(1, dtype=cp.float32)
        probe += probe
        _sync()
        return True, ""
    except Exception as exc:
        return False, _brief(exc)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.list_ops:
        _print_list(args)
        return 0

    if cp is None:
        print(f"cupy 导入失败, 无法基准测试: {_brief(_CUPY_IMPORT_ERROR)}",
              file=sys.stderr)
        print("提示: 用 `python benchmark.py --list` 查看算子矩阵 (不需要设备)。",
              file=sys.stderr)
        return 2

    available, reason = device_probe()
    if not available:
        print(f"没有可用的 NPU 设备, 无法测试: {reason}", file=sys.stderr)
        print("提示: 用 `python benchmark.py --list` 查看算子矩阵; "
              "本机仅能验证到 L3 (编译 + import + 算子注册)。", file=sys.stderr)
        return 3

    cases = _selected_cases(args)
    _print_header(args)
    print(f"准备测试数据并预热 {len(cases)} 个用例 ... (首次运行可能较慢)")
    cache = DataCache(args.vector_size, args.matrix_size)

    results: list[Result] = []
    last_category = None
    try:
        for category, spec, dtype_label, where in cases:
            if category != last_category:
                if last_category is not None:
                    print()
                last_category = category
            row = run_case(spec, dtype_label, where, cache, args)
            row.category = category
            results.append(row)
            _print_result_row(row)
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n中断, 输出已完成部分的结果。", file=sys.stderr)

    code = _print_summary(results, args) if results else 0
    if args.csv and results:
        _write_csv(args.csv, results)
    return code


if __name__ == "__main__":
    sys.exit(main())
