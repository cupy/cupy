"""NumPy/CuPy -> Ascend compatibility migration analyzer (v0.1)."""

from .models import (APIUsage, Category, Diagnostic, Fix, FixMode,
                     Replacement, Severity, SourceLocation, Status)

__version__ = "0.1.0"

__all__ = [
    "APIUsage", "Category", "Diagnostic", "Fix", "FixMode", "Replacement",
    "Severity", "SourceLocation", "Status", "__version__",
]
