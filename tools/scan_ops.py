#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""numpy-ascend CST 数据库构建器 (tree-sitter).

为什么需要它
------------
`Progress.md` / `docs/ascend/*.md` 里的算子覆盖率、注册数、CUDA 残留等结论此前都是
**手工 grep 出来再写进文档**的 (plan.md §5 第 5 条: "引入 tree-sitter-cython, 写一个
tools/scan_ops.py 自动生成算子覆盖率报告, 替代手工维护")。本脚本把这件事自动化:

    python tools/scan_ops.py            # 重新扫描 -> 覆盖 tools/cst_db.json / cst_db.md
    python tools/scan_ops.py --check     # 只校验磁盘上的 db 是否过期 (CI 用)

扫描结果 (CST 事实, 不是正则猜的) 写入两个文件:

    tools/cst_db.json   # 机器可读数据库 (可 diff / 可被其它脚本消费)
    tools/cst_db.md     # 人类可读报告

数据库内容
----------
1.  `registrations`      `register_acl_ufunc("ascend_X", OP_TYPE, ...)` (acl_utils.pyx)
2.  `custom_kernels`      `py_register_custom_kernel(...)` (AscendC 自定义 kernel)
3.  `direct_dispatch`     `launch_general_func("ascend_X", ...)` 等直接派发点
4.  `cupy_ufuncs`         `create_ufunc('cupy_X', ...)` / `'cupy_' + name` 动态构造点
5.  `coverage`            已注册 - 已声明 ufunc 的差集 (未移植算子清单)
6.  `aclop_wrappers`      C++ 侧 `aclop_X(...)` 定义, 以及它内部调用的 `aclnnXxx`
7.  `aclnn_includes`      `#include "aclnnop/aclnn_X.h"` (含 CANN 头文件总数)
8.  `cuda_residue`        未中性化的 `cupy.cuda` / `cublas` / `cudnn` ... 出现位置
9.  `cann_version_branches`  `IF CUPY_CANN_VERSION > 0:` / `<= 0:` 成对性检查
10. `export_gaps`          `cupy/__init__.py` 里被注释掉 (未导出) 的公共 API
11. `parse_errors`         语法解析失败的文件 (数据质量自检)

语法: `.py` 用 tree-sitter-python, `.pyx/.pxd` 用 tree-sitter-cython,
`.h/.cxx/.cpp` 用 tree-sitter-cpp (缺 grammar 时该部分降级为 `null` 而不是报错)。

退出码: 0 = 正常; 1 = --check 发现 db 过期; 2 = 缺少 grammar / 仓库路径不对。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SCHEMA_VERSION = 1

#: 扫描根目录 (相对 repo)
SCAN_ROOTS = ("cupy", "cupyx", "install", "tests")

#: 排除的路径片段
EXCLUDE_PARTS = ("third_party", ".git", "__pycache__", "build", "dist",
                 "node_modules", ".codebuddy")

PY_EXT = {".py"}
CY_EXT = {".pyx", ".pxd"}
CPP_EXT = {".h", ".hpp", ".cxx", ".cpp", ".cc"}

#: 算子注册中心 / 派发中心
ACL_UTILS = "cupy/backends/ascend/api/acl_utils.pyx"
ASCEND_HEADER_DIR = "cupy/backends/ascend"
CUPY_INIT = "cupy/__init__.py"

#: 宿主端组合实现 (REQUIRED_OPS 声明了它依赖哪些已注册算子)
COMPOSITE_MODULE = "cupy/_core/_ascend/composite.py"

#: ufunc 名字的声明位置 (与 skill 的 check_ufunc_coverage.py 保持一致, 便于对比)。
#: 只扫目录顶层: `_core/_gpu`(CUDA 后端内部核) 与 `_core/_ascend`(Ascend 覆写) 里的
#: `cupy_*` 多为内核管道名, 计入覆盖率会引入噪声。
UFUNC_DIRS = ("cupy/_core", "cupy/_math")

#: 动态构造 ufunc 名的 helper: 第一个字符串字面量参数就是 op 名
DYNAMIC_HELPERS = ("create_arithmetic", "create_comparison", "create_bit_op",
                   "_create_bit_op")

#: 内核管道名 (不算用户可见算子) 的形态
INTERNAL_PATTERNS = (
    re.compile(r"^scatter_"),
    re.compile(r".*_kernel$"),
    re.compile(r".*_kernel_"),
    re.compile(r".*_with_dtype$"),
    re.compile(r"^var_core_"),
    re.compile(r"^nan_core_"),
    re.compile(r"^bsum_"),
    re.compile(r"^scan_"),
    re.compile(r"^add_scan_"),
    re.compile(r"^inclusive_batch_scan"),
    re.compile(r"^(hamming|hanning|kaiser|bartlett|blackman|blackmanharris|"
               r"bohman|chebwin|cosine|exponential|flattop|nuttall|parzen|"
               r"taylor|triang|tukey)$"),
    re.compile(r"^(mat_ptrs|cub_|fix|reduceat|_scatter_)$"),
)

#: 直接派发函数
DIRECT_DISPATCH_FUNCS = ("launch_general_func", "launch_reduction_op",
                         "launch_elementwise_func")

_CUPY_UFUNC_LITERAL = re.compile(r"^cupy_[A-Za-z0-9_]+$")
_ACLNN_INCLUDE = re.compile(r"aclnnop/(aclnn_[A-Za-z0-9_]+\.h)")
_COMMENTED_IMPORT = re.compile(
    r"^#\s*from\s+([A-Za-z_][\w.]*)\s+import\s+([A-Za-z_][\w]*)")

#: CUDA 残留符号 -> 归属归类
CUDA_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("cupy.cuda", re.compile(r"^cupy\.cuda")),
    ("cupy_backends", re.compile(r"^cupy_backends")),
    ("cublas", re.compile(r"cublas", re.I)),
    ("cudnn", re.compile(r"cudnn", re.I)),
    ("cufft", re.compile(r"cufft", re.I)),
    ("curand", re.compile(r"curand", re.I)),
    ("cusolver", re.compile(r"cusolver", re.I)),
    ("cusparse", re.compile(r"cusparse", re.I)),
    ("nvrtc", re.compile(r"nvrtc", re.I)),
    ("cuda_runtime", re.compile(r"^cuda[A-Z_]")),
    ("cuda_macro", re.compile(r"^CUDA_")),
    ("cupyx_cuda", re.compile(r"^cupyx\.cuda")),
)

#: 内核管道 / 内部 helper, 不算用户可见算子 (缺失报告里的噪声)
INTERNAL_NAMES = {
    "scatter_add", "scatter_add_mask", "scatter_and", "scatter_and_mask",
    "scatter_max", "scatter_max_mask", "scatter_min", "scatter_min_mask",
    "scatter_or", "scatter_or_mask", "scatter_sub", "scatter_update",
    "scatter_update_mask", "scatter_xor", "scatter_xor_mask",
    "nonzero_kernel", "nonzero_kernel_incomplete_scan", "getitem_mask",
    "prepare_array_indexing", "replace_nan", "put_clip", "put_raise",
    "put_wrap", "choose_clip", "concatenate_same_size", "count_non_nan",
    "mean_empty", "pickup_median", "round_neg_uint", "nan_to_num_",
    "sum_with_dtype", "prod_with_dtype", "nanprod_with_dtype",
    "nanprod_complex_dtype", "nansum_with_dtype", "nansum_complex_dtype",
    "var_core_out", "var_core_float16", "var_core_float32",
    "var_core_float64", "interp", "take_scalar", "searchsorted_kernel",
    "scatter_add_mask_", "fill",
}

INPLACE_PREFIX = "inplace_"

#: 每个 pattern 最多记录多少个样本文件
MAX_SAMPLE_FILES = 10

#: 这些目录本来就是 CUDA/CUDA-like 后端实现, 不算"残留"
RESIDUE_EXPECTED_PREFIXES = (
    "cupy/cuda/",
    "cupy/backends/cuda/",
    "cupy/backends/rocm/",
    "cupy/backends/hip/",
    "cupy/_core/_gpu/",
    "cupyx/scipy/_cuda/",
    "cupy/_environment.py",
)


def is_internal(name: str) -> bool:
    """内核管道 / 非公共 API 的 ufunc 名 -> 不算移植缺口。"""
    if name in INTERNAL_NAMES:
        return True
    return any(pattern.match(name) for pattern in INTERNAL_PATTERNS)


# ---------------------------------------------------------------------------
# tree-sitter 装载
# ---------------------------------------------------------------------------
class Grammars:
    """按需装载 tree-sitter grammar; 缺包时该语言记为不可用。"""

    SPECS = {
        "python": ("tree_sitter_python", PY_EXT),
        "cython": ("tree_sitter_cython", CY_EXT),
        "cpp": ("tree_sitter_cpp", CPP_EXT),
    }

    def __init__(self) -> None:
        self._parsers: dict[str, Any] = {}
        self.versions: dict[str, str] = {}
        self.missing: list[str] = []

    def parser(self, lang: str) -> Optional[Any]:
        if lang in self._parsers:
            return self._parsers[lang]
        module_name = self.SPECS[lang][0]
        parser = None
        try:
            from tree_sitter import Language, Parser
            module = importlib.import_module(module_name)
            with warnings.catch_warnings():
                # tree-sitter-cython 的 language() 返回 int (旧式), 会被
                # tree_sitter>=0.26 判为 deprecated; 这里只需结果, 不刷屏。
                warnings.simplefilter("ignore", DeprecationWarning)
                language = Language(module.language())
            parser = Parser(language)
            self.versions[lang] = _module_version(module_name)
        except Exception as exc:                       # pragma: no cover
            self.missing.append(f"{lang}: {type(exc).__name__}: {exc}")
        self._parsers[lang] = parser
        return parser

    def language_of(self, path: Path) -> Optional[str]:
        suffix = path.suffix
        for lang, (_, extensions) in self.SPECS.items():
            if suffix in extensions:
                return lang
        return None


def _module_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
        for attr in ("__version__", "version"):
            value = getattr(module, attr, None)
            if isinstance(value, str):
                return value
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version(module_name.replace("_", "-"))
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# CST 小工具
# ---------------------------------------------------------------------------
def _is_cython_generated(source: bytes) -> bool:
    """Cython 生成的 .cpp 是构建产物, 解析它只会淹没有用信息。"""
    head = source[:400]
    return b"Generated by Cython" in head


def walk(node) -> Iterator[Any]:
    """前序遍历。"""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def text(node) -> str:
    return node.text.decode("utf-8", "replace")


def string_value(node) -> Optional[str]:
    """字符串字面量的内容 (去掉引号/前缀); 非字符串返回 None。"""
    if node.type not in ("string", "string_literal", "raw_string_literal",
                         "concatenated_string"):
        return None
    raw = text(node)
    match = re.match(r"^[A-Za-z]{0,2}(['\"])(.*)\1$", raw, re.S)
    return match.group(2) if match else None


def is_docstring(node) -> bool:
    """独立成句的字符串 -> docstring (文档里提到 cupy_xxx 不算声明)。"""
    parent = node.parent
    return parent is not None and parent.type == "expression_statement"


def call_name(node) -> Optional[str]:
    """`call` 节点的函数名 (取最后一段, 如 `cupy.add` -> `add`)。"""
    if node.type not in ("call", "call_expression"):
        return None
    function = node.child_by_field_name("function")
    if function is None:
        return None
    raw = text(function)
    return raw.split(".")[-1].strip()


def call_args(node) -> list:
    args = node.child_by_field_name("arguments")
    if args is None:
        return []
    return [child for child in args.children
            if child.type not in ("(", ")", ",", "comment")]


def function_def_name(node) -> Optional[str]:
    """C/C++ `function_definition` 的名字。"""
    if node.type != "function_definition":
        return None
    for child in node.children:
        if child.type == "function_declarator":
            for item in child.children:
                if item.type in ("identifier", "field_identifier",
                                 "qualified_identifier", "destructor_name"):
                    return text(item).split("::")[-1]
    return None


# ---------------------------------------------------------------------------
# 扫描器
# ---------------------------------------------------------------------------
class Scanner:
    def __init__(self, repo: Path, with_runtime: bool = False) -> None:
        self.repo = repo
        self.grammars = Grammars()
        self.with_runtime = with_runtime
        self.parse_errors: list[dict] = []
        self.file_stats: dict[str, int] = {}
        self._trees: dict[Path, Any] = {}

    # -- 基础 --------------------------------------------------------------
    def iter_files(self, extensions: set[str],
                   roots: Iterable[str] = SCAN_ROOTS) -> Iterator[Path]:
        for root in roots:
            base = self.repo / root
            if not base.exists():
                continue
            for path in sorted(base.rglob("*")):
                if not path.is_file() or path.suffix not in extensions:
                    continue
                if any(part in EXCLUDE_PARTS for part in path.parts):
                    continue
                if path.name.endswith(".pyx.cpp"):
                    continue
                yield path

    def tree(self, path: Path):
        """解析并缓存; 记录语法错误。"""
        if path in self._trees:
            return self._trees[path]
        lang = self.grammars.language_of(path)
        parser = self.grammars.parser(lang) if lang else None
        tree = None
        if parser is not None:
            try:
                source = path.read_bytes()
                if _is_cython_generated(source):
                    self._trees[path] = None
                    return None
                tree = parser.parse(source)
            except Exception as exc:                   # pragma: no cover
                self.parse_errors.append({"file": self.rel(path), "error":
                                          f"{type(exc).__name__}: {exc}"})
        if tree is not None:
            self.file_stats[lang] = self.file_stats.get(lang, 0) + 1
            if tree.root_node.has_error:
                self.parse_errors.append({
                    "file": self.rel(path),
                    "error": "tree-sitter reported syntax errors",
                    "lines": self.error_lines(tree.root_node),
                })
        self._trees[path] = tree
        return tree

    def rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.repo))
        except ValueError:
            return str(path)

    @staticmethod
    def error_lines(root, limit: int = 5) -> list[int]:
        lines = []
        for node in walk(root):
            if node.is_error or node.is_missing:
                lines.append(node.start_point[0] + 1)
                if len(lines) >= limit:
                    break
        return lines

    def source_files(self) -> list[Path]:
        return list(self.iter_files(PY_EXT | CY_EXT))

    # -- 1. 算子注册 -------------------------------------------------------
    def scan_registrations(self) -> list[dict]:
        path = self.repo / ACL_UTILS
        tree = self.tree(path) if path.is_file() else None
        if tree is None:
            return []
        found: dict[tuple[str, str], dict] = {}
        for node in walk(tree.root_node):
            name = call_name(node)
            if name not in ("register_acl_ufunc", "register_acl_ufunc_alias"):
                continue
            args = call_args(node)
            if len(args) < 2:
                continue
            opname = string_value(args[0]) or text(args[0]).strip('"')
            if not opname.startswith("ascend_"):
                continue
            key = (opname, text(args[1]))
            entry = found.setdefault(key, {
                "op": opname, "op_type": text(args[1]),
                "line": node.start_point[0] + 1, "count": 0,
            })
            entry["count"] += 1
        return sorted(found.values(), key=lambda e: (e["op"], e["op_type"]))

    def scan_custom_kernels(self) -> list[dict]:
        """AscendC 自定义 kernel 注册表 `CUSTOM_UFUNCS = {...}` (dict 字面量的键)。

        这些名字在 import 时以 `'ascend_' + name` 注册进 `_custom_kernel_specs`,
        属于**独立注册表**, 不在 `py_list_acl_ufuncs()` 里 —— 所以必须单独扫描,
        否则会被误报成"未移植"。
        """
        out: list[dict] = []
        for path in self.iter_files(PY_EXT | CY_EXT):
            tree = self.tree(path)
            if tree is None:
                continue
            for node in walk(tree.root_node):
                if node.type not in ("assignment", "expression_statement"):
                    continue
                target = node.child_by_field_name("left")
                if target is None or text(target).strip() != "CUSTOM_UFUNCS":
                    continue
                value = node.child_by_field_name("right")
                if value is None or value.type != "dictionary":
                    continue
                for pair in value.children:
                    if pair.type != "pair":
                        continue
                    key = pair.child_by_field_name("key")
                    name = string_value(key) if key is not None else None
                    if not name:
                        continue
                    pair_text = text(pair)
                    entry = re.search(r"'entry'\s*:\s*'([^']+)'", pair_text)
                    n_out = re.search(r"'n_out'\s*:\s*(\d+)", pair_text)
                    n_in = re.search(r"'n_in'\s*:\s*(\d+)", pair_text)
                    out.append({
                        "op": name,
                        "file": self.rel(path),
                        "line": pair.start_point[0] + 1,
                        "kernel": entry.group(1) if entry else None,
                        "n_out": int(n_out.group(1)) if n_out else None,
                        "n_in": int(n_in.group(1)) if n_in else None,
                    })
        return sorted(out, key=lambda item: item["op"])

    def scan_host_composites(self) -> list[dict]:
        """宿主端组合实现（`cupy/_core/_ascend/composite.py` 的 `REQUIRED_OPS`）。

        这些算子在 Ascend 上没有 aclnn 实现，但公开 API 在 `is_ascend()` 分支里
        改走"用已注册算子拼出来"的路径（`nanargmax`/`nanmean`/`choose`/`angle_deg`…）。
        它们的名字不会出现在 `_builtin_operators` 里，所以必须单独扫描，
        否则会被误报成"未移植"；同时校验它依赖的算子确实都有注册。
        """
        path = self.repo / COMPOSITE_MODULE
        tree = self.tree(path) if path.is_file() else None
        if tree is None:
            return []
        for node in walk(tree.root_node):
            if node.type not in ("assignment", "expression_statement"):
                continue
            target = node.child_by_field_name("left")
            if target is None or text(target).strip() != "REQUIRED_OPS":
                continue
            value = node.child_by_field_name("right")
            if value is None or value.type != "dictionary":
                continue
            out: list[dict] = []
            for pair in value.children:
                if pair.type != "pair":
                    continue
                key = pair.child_by_field_name("key")
                name = string_value(key) if key is not None else None
                if not name:
                    continue
                # 只解析 value (key 就是算子名本身, 不能算依赖)
                deps_node = pair.child_by_field_name("value")
                deps = sorted(set(re.findall(
                    r"'([a-z0-9_]+)'",
                    text(deps_node) if deps_node is not None else "")))
                out.append({
                    "op": name,
                    "file": self.rel(path),
                    "line": pair.start_point[0] + 1,
                    "requires": deps,
                })
            return sorted(out, key=lambda item: item["op"])
        return []

    # -- 2. 直接派发 -------------------------------------------------------
    def scan_direct_dispatch(self) -> list[dict]:
        out: list[dict] = []
        for path in self.iter_files(CY_EXT | PY_EXT):
            tree = self.tree(path)
            if tree is None:
                continue
            for node in walk(tree.root_node):
                if call_name(node) not in DIRECT_DISPATCH_FUNCS:
                    continue
                args = call_args(node)
                if not args:
                    continue
                opname = string_value(args[0])
                if not opname or not opname.startswith("ascend_"):
                    continue
                out.append({"op": opname, "file": self.rel(path),
                            "line": node.start_point[0] + 1,
                            "via": call_name(node)})
        return sorted(out, key=lambda e: (e["op"], e["file"], e["line"]))

    # -- 3. cupy ufunc 声明 ------------------------------------------------
    def scan_cupy_ufuncs(self) -> dict:
        literals: dict[str, dict] = {}
        dynamic_sites: list[dict] = []

        def record(name: str, path: Path) -> None:
            rel = self.rel(path)
            entry = literals.setdefault(name, {"name": f"cupy_{name}",
                                               "files": [], "count": 0})
            entry["count"] += 1
            if rel not in entry["files"]:
                entry["files"].append(rel)

        for directory in UFUNC_DIRS:
            base = self.repo / directory
            if not base.exists():
                continue
            for path in sorted(base.glob("*")):
                if not path.is_file() or path.suffix not in (PY_EXT | CY_EXT):
                    continue
                tree = self.tree(path)
                if tree is None:
                    continue
                for node in walk(tree.root_node):
                    # (a) 字面量 'cupy_xxx'
                    value = string_value(node)
                    if value and not is_docstring(node):
                        if _CUPY_UFUNC_LITERAL.match(value):
                            record(value[len("cupy_"):], path)
                    # (b) 动态构造 'cupy_' + name / OP_PREFIX + name
                    if node.type == "binary_operator" and "+" in text(node) \
                            and ("'cupy_'" in text(node)
                                 or "OP_PREFIX" in text(node)):
                        rel = self.rel(path)
                        if not any(s["file"] == rel for s in dynamic_sites):
                            dynamic_sites.append({"file": rel,
                                                  "line": node.start_point[0] + 1,
                                                  "text": text(node)})
                    # (c) helper 调用点的第一个字面量参数 = 动态名字
                    if call_name(node) in DYNAMIC_HELPERS:
                        args = call_args(node)
                        if not args:
                            continue
                        raw_name = string_value(args[0])
                        if not raw_name:
                            continue
                        name = (raw_name[len("cupy_"):]
                                if raw_name.startswith("cupy_") else raw_name)
                        if re.match(r"^[a-z_][a-z0-9_]*$", name):
                            record(name, path)
        return {
            "scope": list(UFUNC_DIRS),
            "names": {key: literals[key] for key in sorted(literals)},
            "dynamic_prefix_sites": dynamic_sites,
        }

    # -- 4. C++ 包装 -------------------------------------------------------
    def scan_aclop_wrappers(self) -> dict:
        wrappers: dict[str, dict] = {}
        includes: dict[str, list[str]] = {}
        for path in self.iter_files(CPP_EXT, roots=(ASCEND_HEADER_DIR,)):
            tree = self.tree(path)
            if tree is None:
                continue
            rel = self.rel(path)
            for node in walk(tree.root_node):
                if node.type == "preproc_include":
                    match = _ACLNN_INCLUDE.search(text(node))
                    if match:
                        includes.setdefault(match.group(1), []).append(rel)
                if node.type == "function_definition":
                    name = function_def_name(node)
                    if name and name.startswith("aclop_"):
                        aclnn = sorted({
                            text(c).strip()
                            for c in walk(node)
                            if c.type == "identifier"
                            and text(c).startswith("aclnn")})
                        entry = wrappers.setdefault(name, {
                            "wrapper": name, "file": rel,
                            "line": node.start_point[0] + 1, "aclnn": [],
                        })
                        for item in aclnn:
                            if item not in entry["aclnn"]:
                                entry["aclnn"].append(item)
        cann_headers = self.count_cann_headers()
        return {
            "wrappers": {key: wrappers[key] for key in sorted(wrappers)},
            "aclnn_includes": {key: {"files": sorted(set(value))}
                               for key, value in sorted(includes.items())},
            "cann_headers_available": cann_headers,
        }

    def count_cann_headers(self) -> Optional[int]:
        home = os.environ.get("ASCEND_HOME_PATH")
        if not home:
            return None
        candidate = Path(home) / "include" / "aclnnop"
        if not candidate.is_dir():
            return None
        return len(list(candidate.glob("aclnn_*.h")))

    # -- 5. CUDA 残留 ------------------------------------------------------
    def scan_cuda_residue(self) -> dict:
        """CUDA 符号残留。分两组:

        * `all`            —— 全仓统计 (含 `cupy/cuda/*` 这类**本来就应该**是 CUDA 的目录)
        * `need_neutralize` —— 只统计非 CUDA 目录 (`cupy/_core`, `cupy/xpu`,
          `cupy/backends/ascend`, ...), 即 plan.md §4.4 要清理的对象
        """
        buckets: dict[str, dict] = {}
        neutral: dict[str, dict] = {}
        for path in self.source_files():
            tree = self.tree(path)
            if tree is None:
                continue
            rel = self.rel(path)
            is_neutral = not rel.startswith(RESIDUE_EXPECTED_PREFIXES)
            for node in walk(tree.root_node):
                if node.type not in ("identifier", "attribute",
                                     "dotted_name", "import_from_statement",
                                     "module_alias"):
                    continue
                raw = text(node)
                for bucket, pattern in CUDA_PATTERNS:
                    if not pattern.search(raw):
                        continue
                    self._add_residue(buckets, bucket, pattern.pattern, rel,
                                      node, raw)
                    if is_neutral:
                        self._add_residue(neutral, bucket, pattern.pattern,
                                          rel, node, raw)
                    break
        return {
            "all": self._finalize_residue(buckets),
            "need_neutralize": self._finalize_residue(neutral),
        }

    @staticmethod
    def _add_residue(buckets: dict, bucket: str, pattern: str, rel: str,
                     node, raw: str) -> None:
        entry = buckets.setdefault(bucket, {
            "pattern": pattern, "count": 0, "per_file": {}, "top_files": [],
            "samples": [],
        })
        entry["count"] += 1
        entry["per_file"][rel] = entry["per_file"].get(rel, 0) + 1
        if len(entry["samples"]) < 5:
            sample = f"{rel}:{node.start_point[0] + 1} {raw[:60]}"
            if sample not in entry["samples"]:
                entry["samples"].append(sample)

    @staticmethod
    def _finalize_residue(buckets: dict) -> dict:
        """把 per_file 汇总成 file_total + top_files (按出现次数排序)。"""
        out: dict[str, dict] = {}
        for bucket, entry in buckets.items():
            per_file = entry.pop("per_file", {})
            top = sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0]))
            entry["file_total"] = len(per_file)
            entry["top_files"] = [{"file": name, "count": count}
                                  for name, count in top[:MAX_SAMPLE_FILES]]
            out[bucket] = entry
        return {key: out[key] for key in sorted(out)}

    # -- 6. IF CUPY_CANN_VERSION 分支 --------------------------------------
    def scan_cann_version_branches(self) -> list[dict]:
        out: list[dict] = []
        for path in self.iter_files(CY_EXT | PY_EXT):
            tree = self.tree(path)
            if tree is None:
                continue
            for node in walk(tree.root_node):
                if node.type not in ("IF_statement", "if_statement"):
                    continue
                if "CUPY_CANN_VERSION" not in text(node):
                    continue
                children = [c for c in node.children
                            if c.type not in ("IF", "if", ":", "block")]
                out.append({
                    "file": self.rel(path),
                    "line": node.start_point[0] + 1,
                    "condition": text(children[0]) if children else "?",
                    "has_else": any(c.type == "ELSE" for c in node.children),
                })
        return sorted(out, key=lambda e: (e["file"], e["line"]))

    # -- 7. cupy/__init__.py 导出缺口 --------------------------------------
    def scan_export_gaps(self) -> dict:
        path = self.repo / CUPY_INIT
        tree = self.tree(path) if path.is_file() else None
        if tree is None:
            return {}
        exported: set[str] = set()
        commented: list[dict] = []
        for node in walk(tree.root_node):
            if node.type in ("import_from_statement", "import_from"):
                raw = text(node)
                match = re.match(r"from\s+([\w.]+)\s+import\s+(.+)", raw, re.S)
                if not match:
                    continue
                for item in match.group(2).split(","):
                    item = item.split(" as ")[0].strip()
                    if item and item != "*":
                        exported.add(item)
            elif node.type == "comment":
                match = _COMMENTED_IMPORT.match(text(node).strip())
                if match:
                    commented.append({"module": match.group(1),
                                      "name": match.group(2),
                                      "line": node.start_point[0] + 1})
        return {
            "exported_count": len(exported),
            "exported": sorted(exported),
            "commented_out_count": len(commented),
            "commented_out": commented,
        }

    # -- 8. 运行时注册表 (可选交叉校验) ------------------------------------
    def scan_runtime_registry(self) -> dict:
        if not self.with_runtime:
            return {"checked": False}
        try:
            import subprocess
            code = ("from cupy.backends.ascend.api.acl_utils import "
                    "py_list_acl_ufuncs;import json;"
                    "print(json.dumps(sorted(o for o,_ in "
                    "py_list_acl_ufuncs())))")
            proc = subprocess.run([sys.executable, "-c", code],
                                  capture_output=True, text=True, timeout=180,
                                  cwd=str(self.repo))
            line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            names = json.loads(line)
            # 同一算子按 OpType 可重复注册, 所以 entries >= unique
            return {"checked": True, "entries": len(names),
                    "unique": len(set(names)), "names": sorted(set(names)),
                    "note": "来自 py_list_acl_ufuncs(); 不含 AscendC 自定义 kernel"}
        except Exception as exc:
            return {"checked": True, "error": f"{type(exc).__name__}: {exc}"}

    # -- 汇总 --------------------------------------------------------------
    def scan(self) -> dict:
        registrations = self.scan_registrations()
        ufuncs = self.scan_cupy_ufuncs()
        custom_kernels = self.scan_custom_kernels()
        custom_names = {entry["op"] for entry in custom_kernels}
        registered = sorted({entry["op"][len("ascend_"):]
                             for entry in registrations})
        public_registered = [n for n in registered
                             if not n.startswith(INPLACE_PREFIX)]
        declared = sorted(ufuncs["names"])
        covered_builtin = set(public_registered) & set(declared)
        missing = sorted(set(declared) - set(public_registered))
        # 自定义 AscendC kernel 覆盖的算子走独立注册表, 不算缺口
        covered_custom = sorted(set(missing) & custom_names)
        # 宿主端组合实现同理: 公开 API 在 is_ascend() 分支里改走组合路径
        host_composites = self.scan_host_composites()
        coverable = {entry["op"] for entry in host_composites}
        registered_all = set(registered) | custom_names
        for entry in host_composites:
            entry["missing_deps"] = sorted(
                f"ascend_{dep}" for dep in entry["requires"]
                if dep not in registered_all)
        covered_host = sorted(set(missing) & coverable)
        remaining = [n for n in missing
                     if n not in custom_names and n not in coverable]
        actionable = [n for n in remaining if not is_internal(n)]
        internal = [n for n in remaining if is_internal(n)]
        covered = sorted(covered_builtin | set(covered_custom)
                         | set(covered_host))
        wrapper_info = self.scan_aclop_wrappers()
        runtime = self.scan_runtime_registry()

        db = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _dt.datetime.now().astimezone().isoformat(
                timespec="seconds"),
            "generator": "tools/scan_ops.py (tree-sitter CST)",
            "repo": str(self.repo),
            "grammars": {
                "versions": self.grammars.versions,
                "unavailable": self.grammars.missing,
            },
            "scan": {
                "roots": list(SCAN_ROOTS),
                "parsed_files": self.file_stats,
                "parse_errors": self.parse_errors,
            },
            "registrations": registrations,
            "registered_ops": registered,
            "registered_public": public_registered,
            "custom_kernels": custom_kernels,
            "host_composites": host_composites,
            "direct_dispatch": self.scan_direct_dispatch(),
            "cupy_ufuncs": ufuncs,
            "coverage": {
                "registered_total": len(registered),
                "registered_public": len(public_registered),
                "cupy_ufuncs": len(declared),
                "covered": len(covered),
                "covered_list": covered,
                "covered_percent": round(100.0 * len(covered) / len(declared), 1)
                if declared else 0.0,
                "covered_via_custom_kernel": covered_custom,
                "covered_via_host_composite": covered_host,
                "missing_actionable": actionable,
                "missing_internal": internal,
            },
            "aclop_wrappers": wrapper_info["wrappers"],
            "aclnn_includes": wrapper_info["aclnn_includes"],
            "aclnn_headers_available": wrapper_info["cann_headers_available"],
            "cuda_residue": self.scan_cuda_residue(),
            "cann_version_branches": self.scan_cann_version_branches(),
            "export_gaps": self.scan_export_gaps(),
            "runtime_registry": runtime,
        }
        if runtime.get("checked") and runtime.get("names") is not None:
            # 运行时给的是完整名字 (ascend_xxx), 静态扫描存的是后缀, 需归一化
            runtime_names = {name[len("ascend_"):] if name.startswith("ascend_")
                             else name for name in runtime["names"]}
            static_names = set(registered)
            db["coverage"]["static_vs_runtime"] = {
                "runtime_count": len(runtime_names),
                "only_static": sorted(static_names - runtime_names),
                "only_runtime": sorted(runtime_names - static_names),
            }
        return db


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def render_markdown(db: dict) -> str:
    coverage = db["coverage"]
    residue = db["cuda_residue"]
    lines = [
        "# numpy-ascend CST 数据库",
        "",
        f"> 由 `tools/scan_ops.py` 自动生成 ({db['generated_at']}), "
        "**请勿手工编辑**; 重新生成: `python tools/scan_ops.py`。",
        "",
        "## 1. 概览",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 已注册 ascend 算子 (含 inplace) | {coverage['registered_total']} |",
        f"| 其中 public | {coverage['registered_public']} |",
        f"| 声明的 cupy ufunc (cupy/_core + cupy/_math) | {coverage['cupy_ufuncs']} |",
        f"| 已覆盖 (含自定义 kernel) | {coverage['covered']} "
        f"({coverage['covered_percent']} %) |",
        f"| 其中经 AscendC 自定义 kernel 覆盖 | "
        f"{len(coverage['covered_via_custom_kernel'])} |",
        f"| 未覆盖 (需移植) | {len(coverage['missing_actionable'])} |",
        f"| 未覆盖 (内核噪声, 不算缺口) | {len(coverage['missing_internal'])} |",
        f"| `aclop_*` C++ 包装 | {len(db['aclop_wrappers'])} |",
        f"| `aclnn_*` 头文件 include 数 | {len(db['aclnn_includes'])} |",
        f"| CANN 可用 aclnn 头文件总数 | {db['aclnn_headers_available']} |",
        f"| AscendC 自定义 kernel | {len(db['custom_kernels'])} |",
        f"| `IF CUPY_CANN_VERSION` 分支 | {len(db['cann_version_branches'])} |",
        f"| 解析失败文件 | {len(db['scan']['parse_errors'])} |",
        "",
        "## 2. 未覆盖算子 (actionable)",
        "",
    ]
    missing = coverage["missing_actionable"]
    lines += [f"- `cupy_{name}`" for name in missing] or ["- (无)"]
    lines += ["", "### 2.1 经 AscendC 自定义 kernel 覆盖 (不在 builtin 注册表)",
              ""]
    lines += [f"- `{name}`" for name in coverage["covered_via_custom_kernel"]] \
        or ["- (无)"]

    lines += ["", "### 2.2 经宿主端组合实现覆盖 (用已注册算子拼出来)",
              "", "| 算子 | 依赖的已注册算子 | 缺失依赖 |", "|---|---|---|"]
    for entry in db.get("host_composites", []):
        deps = ", ".join(f"`{d}`" for d in entry["requires"]) or "(无)"
        missing_deps = ", ".join(entry.get("missing_deps", [])) or "—"
        lines.append(f"| `{entry['op']}` | {deps} | {missing_deps} |")
    if not db.get("host_composites"):
        lines.append("| (无) | | |")

    lines += ["", "## 3. CUDA 残留 (CST 统计)", "",
              "### 3.1 需要中性化的路径 (排除 `cupy/cuda`, `cupy/backends/cuda`)",
              "", "| 符号族 | 出现次数 | 文件数 |", "|---|---|---|"]
    for bucket, entry in residue["need_neutralize"].items():
        lines.append(f"| {bucket} | {entry['count']} | {entry['file_total']} |")
    if not residue["need_neutralize"]:
        lines.append("| (无) | 0 | 0 |")
    lines += ["", "热点文件 (按引用次数):", ""]
    for bucket, entry in residue["need_neutralize"].items():
        for item in entry.get("top_files", [])[:5]:
            lines.append(f"- `{bucket}`: {item['file']} ({item['count']})")
    lines += ["", "### 3.2 全仓 (含 CUDA 后端本身)", "",
              "| 符号族 | 出现次数 | 文件数 |", "|---|---|---|"]
    for bucket, entry in residue["all"].items():
        lines.append(f"| {bucket} | {entry['count']} | {entry['file_total']} |")
    if not residue["all"]:
        lines.append("| (无) | 0 | 0 |")

    lines += ["", "## 4. `IF CUPY_CANN_VERSION` 分支", "",
              "| 文件 | 行 | 条件 | 有 ELSE |", "|---|---|---|---|"]
    for entry in db["cann_version_branches"]:
        lines.append(f"| {entry['file']} | {entry['line']} | "
                     f"`{entry['condition']}` | {'yes' if entry['has_else'] else 'no'} |")
    if not db["cann_version_branches"]:
        lines.append("| (无) | | | |")

    gaps = db.get("export_gaps") or {}
    lines += ["", "## 5. `cupy/__init__.py` 顶层导出缺口", "",
              f"- 已导出: {gaps.get('exported_count', 0)}",
              f"- 被注释掉 (未导出): {gaps.get('commented_out_count', 0)}", ""]
    lines += [f"- `{entry['name']}`  <-  {entry['module']} (line {entry['line']})"
              for entry in gaps.get("commented_out", [])] or ["- (无)"]

    runtime = db.get("runtime_registry") or {}
    lines += ["", "## 6. 运行时注册表交叉校验", ""]
    if runtime.get("names") is not None:
        cross = coverage.get("static_vs_runtime", {})
        lines.append(f"- 运行时注册条目 (含同算子不同 OpType): {runtime.get('entries')}")
        lines.append(f"- 去重后算子数: {runtime.get('unique')}")
        lines.append(f"- 仅运行时可见 (CST 漏掉): "
                     f"{cross.get('only_runtime') or '无'}")
        lines.append(f"- 仅静态扫描可见 (运行时缺): "
                     f"{cross.get('only_static') or '无'}")
        lines.append("- 说明: 自定义 AscendC kernel 走 `_custom_kernel_specs`, "
                     "不在此表内, 见 `custom_kernels`")
    elif runtime.get("error"):
        lines.append(f"- 跳过: {runtime['error']}")
    else:
        lines.append("- 跳过 (未加 `--runtime`)")

    lines += ["", "## 7. 数据质量", "",
              f"- grammar: {db['grammars']['versions']}",
              f"- 解析文件数: {db['scan']['parsed_files']}", ""]
    for item in db["scan"]["parse_errors"]:
        lines.append(f"- 语法错误: `{item['file']}` {item.get('lines', '')}")
    if not db["scan"]["parse_errors"]:
        lines.append("- 无语法错误")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
#: 只有 `--runtime` 才会产生的字段: 不带 --runtime 的 --check 不能拿它们比较,
#: 否则磁盘上的库(带 runtime 数据)永远"过期"。
_RUNTIME_ONLY_KEYS = ("runtime_registry", "static_vs_runtime")


def _stable(db: dict, with_runtime: bool = False) -> dict:
    """去掉时间戳等易变字段, 用于 --check 比较。

    Args:
        with_runtime: 本次是否真的查了运行时注册表; 为 False 时把 runtime 相关
            字段一并剔除, 保证"用同一种方式重算"再比较。
    """
    clone = json.loads(json.dumps(db))
    clone.pop("generated_at", None)
    clone.pop("repo", None)
    if not with_runtime:
        for key in _RUNTIME_ONLY_KEYS:
            clone.pop(key, None)
            coverage = clone.get("coverage")
            if isinstance(coverage, dict):
                coverage.pop(key, None)
    return clone


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="构建/刷新 numpy-ascend 的 tree-sitter CST 数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]),
                        help="仓库根目录")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent),
                        help="输出目录 (cst_db.json / cst_db.md)")
    parser.add_argument("--runtime", action="store_true",
                        help="额外导入 cupy 读取运行时算子注册表做交叉校验")
    parser.add_argument("--check", action="store_true",
                        help="只比较磁盘上的 db 是否过期 (不写文件)")
    parser.add_argument("--print-json", action="store_true",
                        help="把 JSON 打到 stdout")
    args = parser.parse_args(argv)

    repo = Path(args.repo).expanduser().resolve()
    if not (repo / ACL_UTILS).is_file():
        print(f"error: {repo / ACL_UTILS} 不存在, --repo 是否正确?", file=sys.stderr)
        return 2

    scanner = Scanner(repo, with_runtime=args.runtime)
    db = scanner.scan()
    if not scanner.grammars.versions:
        print("error: 没有可用的 tree-sitter grammar "
              "(pip install tree-sitter tree-sitter-cython tree-sitter-python "
              "tree-sitter-cpp)", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).expanduser().resolve()
    json_path = out_dir / "cst_db.json"
    md_path = out_dir / "cst_db.md"
    payload = json.dumps(db, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    report = render_markdown(db)

    if args.check:
        if not json_path.is_file():
            print(f"STALE: {json_path} 不存在, 请运行 python tools/scan_ops.py",
                  file=sys.stderr)
            return 1
        old = json.loads(json_path.read_text(encoding="utf-8"))
        if _stable(old, args.runtime) == _stable(db, args.runtime):
            print(f"OK: {json_path.name} 是最新的")
            return 0
        print(f"STALE: {json_path.name} 与当前代码不一致", file=sys.stderr)
        for section in ("coverage", "registrations", "cuda_residue",
                        "cann_version_branches", "export_gaps",
                        "host_composites", "custom_kernels",
                        "aclop_wrappers", "aclnn_includes"):
            old_part = _stable({section: old.get(section)}, args.runtime)
            new_part = _stable({section: db.get(section)}, args.runtime)
            if old_part != new_part:
                print(f"  - 变化的区段: {section}", file=sys.stderr)
        print("  请运行: python tools/scan_ops.py"
              + (" --runtime" if args.runtime else ""), file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(payload, encoding="utf-8")
    md_path.write_text(report, encoding="utf-8")

    coverage = db["coverage"]
    print(f"CST 数据库已更新: {json_path}")
    print(f"                   {md_path}")
    print(f"  parsed files : {db['scan']['parsed_files']}")
    print(f"  registered   : {coverage['registered_total']} "
          f"({coverage['registered_public']} public)")
    print(f"  cupy ufuncs  : {coverage['cupy_ufuncs']} | covered "
          f"{coverage['covered']} ({coverage['covered_percent']} %) | missing "
          f"{len(coverage['missing_actionable'])}")
    print(f"  aclop/ aclnn : {len(db['aclop_wrappers'])} / "
          f"{len(db['aclnn_includes'])}")
    print(f"  IF branches  : {len(db['cann_version_branches'])} | parse errors "
          f"{len(db['scan']['parse_errors'])}")
    if db["grammars"]["unavailable"]:
        print(f"  grammar 缺失 : {db['grammars']['unavailable']}")
    if args.print_json:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
