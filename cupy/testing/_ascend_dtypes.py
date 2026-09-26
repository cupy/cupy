"""Ascend(NPU) 不支持的 dtype 过滤策略 —— 测试收集期共用。

为什么需要
----------
`README.md` §limitation 写明 NPU 的 dtype 限制:

* 全量算子只保证 ``float32``; ``float64`` 仅 add/subtract/multiply/true_divide 可用;
* ``complex128`` 无算子支持; ``complex64`` 已适配, 不再跳过。

若不处理, ``pytest`` 跑上游 CuPy 测试时会因为"dtype 本身不被支持"而大面积 FAIL,
真正的移植缺陷被淹没。本模块把"哪些 dtype 在 Ascend 上不该收集"集中定义一次,
由两处消费:

1. :mod:`cupy.testing._loops` —— ``for_all_dtypes()`` / ``for_float_dtypes()`` 等
   装饰器的候选 dtype 列表在 import 时过滤 (去掉 float64/complex*);
2. ``tests/conftest.py`` —— ``pytest_collection_modifyitems`` 把
   ``@pytest.mark.parametrize('dtype', [numpy.float64, ...])`` 这类显式参数化用例
   标记为 skip (原因是"NPU 不支持该 dtype"), 而不是让它 FAIL。

策略开关
--------
============== ==========================================================
``auto`` (默认) 仅当后端是 Ascend 时过滤; CUDA/CPU 跑完整 dtype 矩阵
``on``          强制过滤 (无 NPU 的开发机上模拟 Ascend 行为)
``off``         完全不过滤 (阶段性放开, 看真实失败)
============== ==========================================================

环境变量
--------
``CUPY_TEST_ASCEND_DTYPE_FILTER``
    ``auto`` / ``on`` / ``off`` (默认 ``auto``)。
``CUPY_TEST_ASCEND_SKIP_DTYPES``
    覆盖默认跳过集合, 逗号分隔, 例如 ``float64`` 只跳过 float64,
    或留空 ``CUPY_TEST_ASCEND_SKIP_DTYPES=`` 表示不跳过任何 dtype。

pytest 侧对应命令行/配置项: ``--ascend-dtype-filter={auto,on,off}`` /
``[tool.pytest.ini_options] ascend_dtype_filter``。
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, Optional, Sequence

import numpy

__all__ = [
    'DEFAULT_SKIP_DTYPES',
    'Policy',
    'configure',
    'describe',
    'dtype_of',
    'enabled',
    'filter_dtypes',
    'is_ascend',
    'is_skipped',
    'item_dtypes',
    'item_dtypes_from_name',
    'policy',
    'reset_cache',
]

#: Ascend 上默认不收集的 dtype (NPU 无算子, 或仅四则运算可用)
#: complex64 已适配 (aclnn 复数算子可用), 不再默认跳过
DEFAULT_SKIP_DTYPES: tuple[str, ...] = ('float64', 'complex128')

_ENV_FILTER = 'CUPY_TEST_ASCEND_DTYPE_FILTER'
_ENV_SKIP = 'CUPY_TEST_ASCEND_SKIP_DTYPES'

_FALSE_WORDS = ('0', 'off', 'false', 'no', 'none')
_TRUE_WORDS = ('1', 'on', 'true', 'yes')

_MODES = ('auto', 'on', 'off')

#: 参数名像 dtype 的 ``@pytest.mark.parametrize`` 项, 其字符串值才按 dtype 解析。
#: (例如 ``parametrize('order', ['C', 'F'])`` 里的 'F' 是排布顺序, 不是 dtype)
_DTYPE_PARAM_NAME_RE = re.compile(r'dtyp|dtype|type|typecode', re.I)

_ASCEND_CACHE: Optional[bool] = None
_SESSION_MODE: Optional[str] = None

#: 从测试 id 里抓 ``dtype=float64`` / ``typecode=c8`` 这类片段
#: (动态生成的测试类把参数写进类名, 没有 callspec, 只能这样解析)
_NODEID_DTYPE_RE = re.compile(
    r"(?<![\w.])(?:\w*(?:dtyp|typecode)\w*|\w*type\w*)\s*=\s*"
    r"([A-Za-z_][\w.]*)")


# ---------------------------------------------------------------------------
# 后端判断 / 模式
# ---------------------------------------------------------------------------
def is_ascend() -> bool:
    """是否 Ascend 后端 (结果缓存; 可在测试里用 :func:`reset_cache` 清掉)。"""
    global _ASCEND_CACHE
    if _ASCEND_CACHE is None:
        override = os.environ.get('CUPY_TEST_ASCEND')
        if override is not None:
            _ASCEND_CACHE = override.lower() not in _FALSE_WORDS
        else:
            try:
                from cupy.backends.backend import is_ascend as _f
                _ASCEND_CACHE = bool(_f)
            except Exception:
                # 未安装 / 未编译时按"非 Ascend"处理, 不影响 CUDA/CPU 测试
                _ASCEND_CACHE = False
    return _ASCEND_CACHE


def reset_cache() -> None:
    """清空缓存 (供测试使用)。"""
    global _ASCEND_CACHE, _SESSION_MODE
    _ASCEND_CACHE = None
    _SESSION_MODE = None


def configure(mode: Optional[str] = None) -> None:
    """设置本次进程的过滤模式 (pytest 在 ``pytest_configure`` 里调用)。

    Args:
        mode: ``'auto'`` / ``'on'`` / ``'off'``; ``None`` 表示清除覆盖。
    """
    global _SESSION_MODE
    if mode is not None:
        mode = str(mode).strip().lower()
        if mode not in _MODES:
            raise ValueError(
                f'ascend dtype filter 模式必须是 {_MODES} 之一, 收到 {mode!r}')
    _SESSION_MODE = mode


def _effective_mode(mode: Optional[str] = None) -> str:
    for candidate in (mode, _SESSION_MODE, os.environ.get(_ENV_FILTER)):
        if candidate:
            candidate = str(candidate).strip().lower()
            if candidate in _FALSE_WORDS:
                return 'off'
            if candidate in _TRUE_WORDS:
                return 'on'
            if candidate in _MODES:
                return candidate
    return 'auto'


def skip_dtype_names() -> tuple[str, ...]:
    """当前要跳过的 dtype 名 (受 ``CUPY_TEST_ASCEND_SKIP_DTYPES`` 影响)。"""
    raw = os.environ.get(_ENV_SKIP)
    if raw is None:
        return DEFAULT_SKIP_DTYPES
    names = []
    for item in raw.replace(';', ',').split(','):
        item = item.strip()
        if not item:
            continue
        try:
            names.append(numpy.dtype(item).name)
        except Exception:
            continue
    return tuple(names)


# ---------------------------------------------------------------------------
# 策略快照
# ---------------------------------------------------------------------------
class Policy:
    """一次 session 内生效的过滤策略 (不可变快照)。

    Attributes:
        enabled: 是否真的过滤。
        mode: 生效模式 (``auto`` / ``on`` / ``off``)。
        names: 被跳过的 dtype 名 (如 ``('float64', 'complex64')``)。
        chars: 被跳过的 dtype 字符码, 用于快速比较。
    """

    __slots__ = ('enabled', 'mode', 'names', 'chars')

    def __init__(self, enabled: bool, mode: str, names: Sequence[str] = ()):
        self.enabled = bool(enabled)
        self.mode = mode
        self.names = tuple(names)
        self.chars = frozenset(
            numpy.dtype(name).char for name in self.names)

    # -- 判断 ------------------------------------------------------------
    def is_skipped(self, dtype: Any) -> bool:
        """给定 dtype (类型 / ``numpy.dtype`` / 名称字符串) 是否被跳过。"""
        if not self.enabled:
            return False
        parsed = dtype_of(dtype)
        if parsed is None:
            return False
        return parsed.char in self.chars

    def to_drop(self, dtypes: Iterable[Any]) -> list:
        return [d for d in dtypes if self.is_skipped(d)]

    def filter(self, dtypes: Iterable[Any]) -> tuple:
        """去掉被跳过的 dtype, 保持原顺序; 未启用时原样返回。"""
        if not self.enabled:
            return tuple(dtypes)
        return tuple(d for d in dtypes if not self.is_skipped(d))

    # -- 报告 ------------------------------------------------------------
    def describe(self) -> str:
        if not self.enabled:
            return (f'ascend dtype filter: off (mode={self.mode}) '
                    '-> 跑完整 dtype 矩阵')
        return (f'ascend dtype filter: on (mode={self.mode}) -> 跳过 '
                + ', '.join(self.names))

    def skip_reason(self, hits: Sequence[Any]) -> str:
        names = ', '.join(sorted({numpy.dtype(h).name for h in hits}))
        return (f'Ascend NPU 不支持 {names} (见 README.md §limitation); '
                '用 --ascend-dtype-filter=off 可强制运行')

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f'Policy(enabled={self.enabled}, mode={self.mode!r}, names={self.names})'


def policy(mode: Optional[str] = None) -> Policy:
    """构造当前策略快照。

    Args:
        mode: 显式覆盖模式 (优先于 :func:`configure` 与环境变量)。
    """
    effective = _effective_mode(mode)
    if effective == 'off':
        return Policy(False, 'off', ())
    if effective == 'on':
        return Policy(True, 'on', skip_dtype_names())
    names = skip_dtype_names()
    return Policy(is_ascend() and bool(names), 'auto', names)


# ---------------------------------------------------------------------------
# 便捷函数 (给 cupy.testing._loops 用)
# ---------------------------------------------------------------------------
def enabled() -> bool:
    return policy().enabled


def filter_dtypes(dtypes: Iterable[Any]) -> tuple:
    return policy().filter(dtypes)


def is_skipped(dtype: Any) -> bool:
    return policy().is_skipped(dtype)


def describe() -> str:
    return policy().describe()


# ---------------------------------------------------------------------------
# dtype 解析 / pytest 项提取
# ---------------------------------------------------------------------------
def dtype_of(value: Any) -> Optional[numpy.dtype]:
    """尽力把 ``value`` 解析成 ``numpy.dtype``; 解析不了返回 ``None``。"""
    if value is None:
        return None
    if isinstance(value, numpy.dtype):
        return value
    if isinstance(value, numpy.generic):
        return value.dtype
    if isinstance(value, type) and issubclass(value, numpy.generic):
        return numpy.dtype(value)
    if isinstance(value, str):
        # 只接受有意义的类型名 ('float64'), 不接受单字符代码 ('C'/'F' 常是
        # 排布顺序参数) —— 见 _DTYPE_PARAM_NAME_RE 的调用方
        if len(value) < 2:
            return None
        try:
            return numpy.dtype(value)
        except Exception:
            return None
    return None


def _dtypes_in(value: Any, param_name: str = '') -> list:
    if isinstance(value, str):
        if not _DTYPE_PARAM_NAME_RE.search(param_name):
            return []
        parsed = dtype_of(value)
        return [parsed] if parsed is not None else []
    if isinstance(value, (list, tuple, set, frozenset)):
        out: list = []
        for item in value:
            out.extend(_dtypes_in(item, param_name))
        return out
    parsed = dtype_of(value)
    return [parsed] if parsed is not None else []


def item_dtypes_from_name(name: str) -> list:
    """从测试 id / 动态类名里解析 ``dtype=float64`` 这类片段。"""
    out: list = []
    for match in _NODEID_DTYPE_RE.finditer(name):
        parsed = dtype_of(match.group(1))
        if parsed is not None:
            out.append(parsed)
    return out


def item_dtypes(item: Any) -> list:
    """从一个 pytest item 里提取 dtype 参数。

    三个来源 (合并去重):

    1. ``item.callspec.params`` —— ``@pytest.mark.parametrize`` 展开后的值;
       值本身就是 dtype, 或"参数名像 dtype 且值是字符串"才算。
    2. ``item.nodeid`` —— ``cupy.testing._parameterized.product()`` 生成的是
       **动态类**, 参数写进了类名 (如
       ``..._param_2389_{arg1=False, dtype=float64, name='floor_divide'}``),
       没有 callspec, 只能从 id 里解析 ``dtype=float64`` 这类片段。

    因此 ``parametrize('order', ['C', 'F'])``、``parametrize('float64_distances',
    [False, True])`` 这类不会被误判。
    """
    out: list = []
    callspec = getattr(item, 'callspec', None)
    params = getattr(callspec, 'params', None) or {} if callspec else {}
    for name, value in params.items():
        out.extend(_dtypes_in(value, name))
    nodeid = getattr(item, 'nodeid', None)
    if nodeid:
        out.extend(item_dtypes_from_name(nodeid))
    # 去重 (保序)
    unique: list = []
    for dtype in out:
        if dtype not in unique:
            unique.append(dtype)
    return unique
