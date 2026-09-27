"""Core data models for the NumPy/CuPy -> Ascend migration analyzer.

Diagnostic is the user-facing boundary, Capability/rule is the backend
knowledge boundary, APIUsage is the Python-program boundary. They are kept
separate on purpose (see design doc, section 28).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    UNKNOWN = "unknown"


class Status(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    DISCOURAGED = "discouraged"
    CPU_FALLBACK = "cpu_fallback"
    SEMANTIC_DIFFERENCE = "semantic_difference"
    UNKNOWN = "unknown"


class FixMode(str, Enum):
    NONE = "none"
    AUTO_SAFE = "auto_safe"          # applied by --fix
    AUTO_WARNING = "auto_warning"    # applied by --fix --risky only
    MANUAL = "manual"                # never auto-applied


class Category(str, Enum):
    UNSUPPORTED_API = "UNSUPPORTED_API"
    UNSUPPORTED_DTYPE = "UNSUPPORTED_DTYPE"
    UNSUPPORTED_ARGUMENT = "UNSUPPORTED_ARGUMENT"
    UNSUPPORTED_COMBINATION = "UNSUPPORTED_COMBINATION"
    SEMANTIC_DIFFERENCE = "SEMANTIC_DIFFERENCE"
    CPU_FALLBACK = "CPU_FALLBACK"
    DISCOURAGED_DTYPE = "DISCOURAGED_DTYPE"
    PERFORMANCE_WARNING = "PERFORMANCE_WARNING"
    UNKNOWN_DYNAMIC = "UNKNOWN_DYNAMIC"
    UNKNOWN_API = "UNKNOWN_API"


# Rule priority: when several diagnostics hit the same location, the highest
# priority wins and lower-priority noise (dtype warnings on a broken API) is
# suppressed.
PRIORITY = {
    Category.UNSUPPORTED_API: 100,
    Category.UNSUPPORTED_ARGUMENT: 90,
    Category.UNSUPPORTED_DTYPE: 90,
    Category.UNSUPPORTED_COMBINATION: 90,
    Category.SEMANTIC_DIFFERENCE: 80,
    Category.CPU_FALLBACK: 60,
    Category.DISCOURAGED_DTYPE: 40,
    Category.PERFORMANCE_WARNING: 30,
    Category.UNKNOWN_DYNAMIC: 10,
    Category.UNKNOWN_API: 10,
}


@dataclass
class SourceLocation:
    file: str
    line: int          # 1-based
    column: int        # 1-based, char offset
    end_line: Optional[int] = None
    end_column: Optional[int] = None


@dataclass
class Replacement:
    kind: str                       # rename | dtype | code
    target: str
    fix_mode: FixMode = FixMode.MANUAL
    confidence: float = 1.0
    risk: Optional[str] = None


@dataclass
class Fix:
    """A source-range rewrite (never via ast.unparse, per design doc §17)."""
    start: int                      # char offset into original source
    end: int
    new_text: str
    description: str = ""


@dataclass
class APIUsage:
    """A resolved numpy/cupy call site extracted from the AST."""
    backend: Optional[str]          # "numpy" | "cupy" | None (unknown)
    operation: Optional[str]        # dotted op, e.g. "linalg.inv" or method name
    location: SourceLocation
    node: Any = None                # ast.Call
    func_node: Any = None           # ast.Attribute / ast.Name of the callee
    arguments: dict = field(default_factory=dict)  # kw name -> Constant value
    dtype: Optional[str] = None     # canonical dtype, if statically known
    dtype_node: Any = None          # AST node of the dtype expression
    dtype_dynamic: bool = False     # dtype present but not statically resolvable


@dataclass
class Diagnostic:
    rule_id: str
    category: Category
    severity: Severity
    status: Status
    location: SourceLocation
    source_api: Optional[str] = None
    dtype: Optional[str] = None
    argument: Optional[str] = None
    message: str = ""
    suggestion: Optional[str] = None
    fix_mode: FixMode = FixMode.NONE
    fix: Optional[Fix] = None
    replacement: Optional[Replacement] = None
    confidence: float = 1.0
    source_line: str = ""

    @property
    def auto_fixable(self) -> bool:
        return (
            self.fix is not None
            and self.fix_mode in (FixMode.AUTO_SAFE, FixMode.AUTO_WARNING)
        )

    def to_json(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "status": self.status.value,
            "category": self.category.value,
            "location": {
                "file": self.location.file,
                "line": self.location.line,
                "column": self.location.column,
            },
            "api": self.source_api,
            "dtype": self.dtype,
            "argument": self.argument,
            "message": self.message,
            "suggestion": self.suggestion,
            "fix": {
                "mode": self.fix_mode.value,
                "applied_by": (
                    "--fix" if self.fix_mode == FixMode.AUTO_SAFE
                    else "--fix --risky" if self.fix_mode == FixMode.AUTO_WARNING
                    else None
                ) if self.auto_fixable else None,
            },
            "confidence": self.confidence,
        }
