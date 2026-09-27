"""Reporters: text (terminal) and JSON. Diagnostic is the single boundary."""

from __future__ import annotations

from typing import List

from .models import Diagnostic, FixMode, Severity

SEV_LABEL = {
    Severity.ERROR: "ERROR",
    Severity.WARNING: "WARN",
    Severity.INFO: "INFO",
    Severity.UNKNOWN: "UNKNOWN",
}


def summarize(diagnostics: List[Diagnostic], files: int, usages: int,
              parse_failures: int = 0) -> dict:
    errors = sum(1 for d in diagnostics if d.severity == Severity.ERROR)
    warnings = sum(1 for d in diagnostics if d.severity == Severity.WARNING)
    infos = sum(1 for d in diagnostics if d.severity == Severity.INFO)
    unknown = sum(1 for d in diagnostics if d.severity == Severity.UNKNOWN)
    auto_safe = sum(1 for d in diagnostics
                    if d.fix_mode == FixMode.AUTO_SAFE and d.fix)
    auto_warning = sum(1 for d in diagnostics
                       if d.fix_mode == FixMode.AUTO_WARNING and d.fix)
    manual = sum(1 for d in diagnostics
                 if d.fix_mode == FixMode.MANUAL and d.fix)
    return {
        "files": files,
        "apis_analyzed": usages,
        "errors": errors,
        "warnings": warnings,
        "infos": infos,
        "unknown": unknown,
        "fixable": auto_safe + auto_warning,
        "auto_safe": auto_safe,
        "auto_warning": auto_warning,
        "manual": manual,
        "parse_failures": parse_failures,
    }


class TextReporter:
    def render(self, diagnostics: List[Diagnostic], summary: dict) -> str:
        out = []
        for d in diagnostics:
            loc = d.location
            out.append(f"{loc.file}:{loc.line}:{loc.column} "
                       f"{SEV_LABEL[d.severity]:<7} [{d.category.value}] "
                       f"{d.rule_id}")
            if d.source_line:
                out.append(f"    | {d.source_line.strip()}")
            if d.message:
                out.append(f"    {d.message}")
            if d.suggestion:
                out.append(f"    Suggestion: {d.suggestion}")
            if d.auto_fixable:
                how = ("--fix" if d.fix_mode == FixMode.AUTO_SAFE
                       else "--fix --risky")
                out.append(f"    Auto-fix: YES ({how})")
            elif d.fix_mode == FixMode.MANUAL and d.replacement is not None:
                out.append("    Auto-fix: NO (manual review required)")
            else:
                out.append("    Auto-fix: NO")
            if d.confidence < 1.0:
                out.append(f"    Confidence: {d.confidence:.0%} "
                           "(static analysis)")
            out.append("")

        out.append("Summary:")
        out.append(f"  files scanned:  {summary['files']}")
        out.append(f"  APIs analyzed:  {summary['apis_analyzed']}")
        out.append(f"  errors:   {summary['errors']}")
        out.append(f"  warnings: {summary['warnings']}")
        out.append(f"  infos:    {summary['infos']}")
        out.append(f"  fixable:  {summary['fixable']} "
                   f"(auto_safe: {summary['auto_safe']}, "
                   f"auto_warning: {summary['auto_warning']}, "
                   f"manual: {summary['manual']})")
        if summary["parse_failures"]:
            out.append(f"  parse failures: {summary['parse_failures']}")
        return "\n".join(out)


class JsonReporter:
    def render(self, diagnostics: List[Diagnostic], summary: dict,
               db=None) -> dict:
        report = {
            "schema_version": "1.0",
            "tool": {"name": "numpy-ascend-check", "version": "0.1.0"},
            "source": {"backends": ["numpy", "cupy"]},
            "target": {
                "backend": "ascend",
                "version": getattr(db, "target_version", "") if db else "",
            },
            "summary": summary,
            "diagnostics": [d.to_json() for d in diagnostics],
        }
        return report
