"""Import / symbol / alias resolver (design doc §27, prototype §6-7).

Resolves `np`, `cp`, `zeros`, `inv`, assignment aliases (`xp = np`) into a
unified abstract namespace so the analyzer never hardcodes `if name == "np"`.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, Optional

KNOWN_BACKENDS = ("numpy", "cupy")
_BACKEND_BY_ROOT = {"numpy": "numpy", "cupy": "cupy"}


@dataclass
class ResolvedSymbol:
    backend: Optional[str]   # "numpy" | "cupy" | None
    full: str                # e.g. "numpy" or "numpy.linalg.inv"


class SymbolTable:
    def __init__(self):
        self.names: Dict[str, ResolvedSymbol] = {}

    def get(self, name: str) -> Optional[ResolvedSymbol]:
        return self.names.get(name)


def build_symbol_table(tree: ast.AST) -> SymbolTable:
    st = SymbolTable()

    def _bind(local: str, sym: ResolvedSymbol):
        # do not overwrite earlier bindings (first binding wins, like Python
        # at module scope for the common case)
        if local not in st.names:
            st.names[local] = sym

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in _BACKEND_BY_ROOT:
                    local = a.asname or root
                    _bind(local, ResolvedSymbol(_BACKEND_BY_ROOT[root], a.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue
            root = node.module.split(".")[0]
            if root not in _BACKEND_BY_ROOT:
                continue
            for a in node.names:
                if a.name == "*":
                    continue
                local = a.asname or a.name
                _bind(local, ResolvedSymbol(_BACKEND_BY_ROOT[root],
                                            f"{node.module}.{a.name}"))
        elif isinstance(node, ast.Assign):
            # simple alias propagation: xp = np / foo = np.linalg
            v = node.value
            target_full = None
            if isinstance(v, ast.Name) and v.id in st.names:
                target_full = st.names[v.id].full
            elif (isinstance(v, ast.Attribute)
                  and isinstance(v.value, ast.Name)
                  and v.value.id in st.names):
                target_full = st.names[v.value.id].full + "." + v.attr
            if target_full:
                backend = target_full.split(".")[0]
                if backend in _BACKEND_BY_ROOT:
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            _bind(t.id, ResolvedSymbol(backend, target_full))
    return st


def resolve_expr(node: ast.AST, st: SymbolTable) -> Optional[ResolvedSymbol]:
    """Resolve a dotted attribute chain rooted at a known name.

    np.linalg.inv -> ResolvedSymbol("numpy", "numpy.linalg.inv")
    Returns None when the root name is not a known namespace alias.
    """
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    sym = st.get(cur.id)
    if sym is None:
        return None
    if parts:
        return ResolvedSymbol(sym.backend, sym.full + "." + ".".join(reversed(parts)))
    return sym
