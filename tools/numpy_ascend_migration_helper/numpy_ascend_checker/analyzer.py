"""AST usage collector: Python source -> APIUsage model (prototype §8-10)."""

from __future__ import annotations

import ast
from typing import List, Optional, Tuple

from .dtypes import canonical_dtype
from .models import APIUsage, SourceLocation
from .resolver import KNOWN_BACKENDS, SymbolTable, resolve_expr
from .source import SourceIndex


class UsageCollector(ast.NodeVisitor):
    def __init__(self, st: SymbolTable, index: SourceIndex):
        self.st = st
        self.index = index
        self.usages: List[APIUsage] = []
        self.consumed: set = set()      # ids of dtype nodes handled by calls

    # ------------------------------------------------------------------ #
    def visit_Call(self, node: ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            method_name = func.attr
        elif isinstance(func, ast.Name):
            method_name = func.id
        else:
            method_name = None

        sym = resolve_expr(func, self.st)
        if sym is not None and sym.backend in KNOWN_BACKENDS:
            backend = sym.backend
            operation = (sym.full.split(".", 1)[1]
                         if "." in sym.full else None)
        else:
            # method call on an unresolved object, e.g. x.astype(...)
            backend = None
            operation = method_name

        dtype, dtype_node, dynamic = self._extract_dtype(node, operation)

        usage = APIUsage(
            backend=backend,
            operation=operation,
            location=self._loc(func if func is not None else node),
            node=node,
            func_node=func,
            arguments=self._const_arguments(node),
            dtype=dtype,
            dtype_node=dtype_node,
            dtype_dynamic=dynamic,
        )
        self.usages.append(usage)
        self.generic_visit(node)

    # ------------------------------------------------------------------ #
    def _extract_dtype(self, call: ast.Call,
                       operation: Optional[str]) -> Tuple[Optional[str], Optional[ast.AST], bool]:
        """Extract a statically-known dtype from a call.

        Handles dtype=<expr> keyword and the first positional argument of
        astype()/view(). Returns (canonical_dtype, node, is_dynamic).
        """
        node = None
        kw = next((k for k in call.keywords if k.arg == "dtype"), None)
        if kw is not None:
            node = kw.value
        elif operation in ("astype", "view") and call.args:
            node = call.args[0]
        if node is None:
            return None, None, False
        return self._dtype_from_node(node)

    def _dtype_from_node(self, node: ast.AST) -> Tuple[Optional[str], Optional[ast.AST], bool]:
        if isinstance(node, ast.Name):
            # builtin spellings: dtype=object / bool / float ...
            c = canonical_dtype(node.id)
            return (c, node, c is None)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                c = canonical_dtype(node.value)
                if c:
                    return c, node, False
                return None, node, True  # e.g. dtype="S10"
            return None, None, False
        if isinstance(node, ast.Call):
            # np.dtype("float32") / np.dtype(something)
            inner = None
            if node.args:
                inner = node.args[0]
            if (isinstance(inner, ast.Constant)
                    and isinstance(inner.value, str)):
                c = canonical_dtype(inner.value)
                return (c, node, c is None)
            return None, node, True
        if isinstance(node, ast.Attribute):
            sym = resolve_expr(node, self.st)
            if sym is not None and sym.backend in KNOWN_BACKENDS:
                c = canonical_dtype(node.attr)
                if c:
                    self.consumed.add(id(node))
                    return c, node, False
            return None, node, True
        # Name / subscript / anything dynamic
        return None, node, True

    @staticmethod
    def _const_arguments(call: ast.Call) -> dict:
        args = {}
        for k in call.keywords:
            if isinstance(k.value, ast.Constant):
                args[k.arg] = k.value.value
            else:
                args[k.arg] = None
        return args

    def _loc(self, node: ast.AST) -> SourceLocation:
        return SourceLocation(
            file=self.index.filename,
            line=node.lineno,
            column=node.col_offset + 1,
            end_line=getattr(node, "end_lineno", None),
            end_column=(getattr(node, "end_col_offset", None) or 0) + 1
            if getattr(node, "end_col_offset", None) is not None else None,
        )


def collect_standalone_dtype_refs(tree: ast.AST, st: SymbolTable,
                                  consumed: set) -> List[Tuple[ast.AST, str]]:
    """Collect `np.<dtype>` / `cp.<dtype>` references outside of calls
    already analyzed (e.g. x.astype(np.uint64), np.can_cast(x, np.float64)).
    """
    refs = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or id(node) in consumed:
            continue
        sym = resolve_expr(node, st)
        if sym is not None and sym.backend in KNOWN_BACKENDS:
            c = canonical_dtype(node.attr)
            if c:
                refs.append((node, c))
    return refs


def analyze_tree(tree: ast.AST, st: SymbolTable, index: SourceIndex):
    collector = UsageCollector(st, index)
    collector.visit(tree)
    dtype_refs = collect_standalone_dtype_refs(tree, st, collector.consumed)
    return collector.usages, dtype_refs
