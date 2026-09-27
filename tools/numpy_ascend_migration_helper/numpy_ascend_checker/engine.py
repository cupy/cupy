"""Rule engine: APIUsage + rules -> Diagnostics (prototype §12-16)."""

from __future__ import annotations

import ast
from typing import List, Optional

from .dtypes import canonical_dtype
from .models import (APIUsage, Category, Diagnostic, Fix, FixMode, PRIORITY,
                     Replacement, Severity, SourceLocation, Status)
from .resolver import KNOWN_BACKENDS
from .source import SourceIndex


class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


class RuleEngine:
    def __init__(self, db, lang: str = "en", report_unknown_api: bool = False):
        self.db = db
        self.lang = lang if lang in ("en", "zh") else "en"
        self.report_unknown_api = report_unknown_api

    # ------------------------------------------------------------------ #
    def analyze_usage(self, usage: APIUsage, index: SourceIndex) -> List[Diagnostic]:
        diags: List[Diagnostic] = []
        if usage.backend in KNOWN_BACKENDS and usage.operation:
            diags.extend(self._analyze_api(usage, index))
        else:
            # method call on an unresolved object: only dtype checks apply
            if usage.dtype is not None:
                d = self._check_dtype(usage, None, index)
                if d:
                    diags.append(d)
            elif usage.dtype_dynamic:
                diags.append(self._unknown_dtype_diag(usage, index))
        return diags

    def analyze_dtype_ref(self, node: ast.AST, backend: str, index: SourceIndex) -> Optional[Diagnostic]:
        """Standalone dtype reference such as `np.uint64`."""
        dt = canonical_dtype(node.attr)
        if dt is None:
            return None
        usage = APIUsage(
            backend=backend,
            operation=None,
            location=SourceLocation(
                file=index.filename,
                line=node.lineno,
                column=node.col_offset + 1,
            ),
            dtype=dt,
            dtype_node=node,
        )
        return self._check_dtype(usage, None, index)

    # ------------------------------------------------------------------ #
    def _analyze_api(self, usage: APIUsage, index: SourceIndex) -> List[Diagnostic]:
        diags: List[Diagnostic] = []
        rule = self.db.lookup(usage.operation, usage.backend)
        api_name = f"{usage.backend}.{usage.operation}"

        if rule is None:
            if self.report_unknown_api:
                diags.append(self._build(
                    "api.unknown", Category.UNKNOWN_API, Severity.INFO,
                    Status.UNKNOWN, usage, index,
                    message=self._msg("api.unknown", api=api_name)))
            # still run global dtype rules on the call's dtype argument
            if usage.dtype is not None:
                d = self._check_dtype(usage, None, index)
                if d:
                    diags.append(d)
            elif usage.dtype_dynamic:
                diags.append(self._unknown_dtype_diag(usage, index))
            return diags

        support = rule.get("support", {})
        status = support.get("status", "supported")

        # priority: if the API itself is unavailable, dtype/argument warnings
        # are meaningless (design doc §22)
        if status == "unsupported":
            d = self._build(
                "api.unsupported", Category.UNSUPPORTED_API, Severity.ERROR,
                Status.UNSUPPORTED, usage, index,
                message=self._msg("api.unsupported", api=api_name,
                                  reason=support.get("reason", "not implemented")),
                suggestion=self._suggest(rule))
            rep = rule.get("replacement")
            if rep:
                d.replacement = Replacement(
                    kind=rep.get("kind", "rename"),
                    target=rep.get("target", ""),
                    fix_mode=FixMode(rep.get("fix_mode", "manual")))
                d.fix_mode = d.replacement.fix_mode
                if d.replacement.fix_mode in (FixMode.AUTO_SAFE, FixMode.AUTO_WARNING):
                    d.fix = self._rename_fix(usage.func_node, d.replacement.target, index)
            diags.append(d)
            return diags

        if status == "cpu_fallback":
            diags.append(self._build(
                "api.cpu_fallback", Category.CPU_FALLBACK, Severity.WARNING,
                Status.CPU_FALLBACK, usage, index,
                message=self._msg("api.cpu_fallback", api=api_name),
                suggestion=self._suggest(rule)))
        elif status == "semantic_difference":
            diags.append(self._build(
                "api.semantic_difference", Category.SEMANTIC_DIFFERENCE,
                Severity.ERROR, Status.SEMANTIC_DIFFERENCE, usage, index,
                message=self._msg("api.semantic_difference", api=api_name,
                                  reason=support.get("reason", "")),
                suggestion=self._suggest(rule)))

        # explicit semantic notes attached to a supported API
        for s in rule.get("semantic", []):
            diags.append(self._build(
                "api.semantic_difference", Category.SEMANTIC_DIFFERENCE,
                Severity.ERROR, Status.SEMANTIC_DIFFERENCE, usage, index,
                message=self._msg("api.semantic_difference", api=api_name,
                                  reason=s.get("reason", "")),
                suggestion=s.get("suggestion") or s.get("suggest_zh")))

        # argument rules
        for ar in rule.get("argument_rules", []):
            d = self._check_argument(usage, rule, ar, index, api_name)
            if d:
                diags.append(d)

        # dtype rules (only meaningful when the API itself is usable)
        if usage.dtype is not None:
            d = self._check_dtype(usage, rule, index)
            if d:
                diags.append(d)
        elif usage.dtype_dynamic:
            diags.append(self._unknown_dtype_diag(usage, index))

        return diags

    # ------------------------------------------------------------------ #
    def _check_argument(self, usage: APIUsage, rule: dict, ar: dict,
                        index: SourceIndex, api_name: str) -> Optional[Diagnostic]:
        arg = ar.get("argument")
        if arg is None:
            return None
        kw = next((k for k in usage.node.keywords if k.arg == arg), None)
        if kw is None:
            return None
        reason = ar.get("reason", "")
        suggestion = ar.get("suggestion") or ar.get("suggest_zh")

        if ar.get("presence_unsupported"):
            return self._build(
                "arg.presence_unsupported", Category.UNSUPPORTED_ARGUMENT,
                Severity.ERROR, Status.UNSUPPORTED, usage, index,
                location=self._loc(kw.value, index),
                argument=arg, api_name=api_name,
                message=self._msg("arg.presence_unsupported", api=api_name,
                                  argument=arg, reason=reason),
                suggestion=suggestion)

        val = kw.value
        if isinstance(val, ast.Constant) and val.value is not None:
            v = val.value
            if v in ar.get("unsupported_values", []):
                return self._build(
                    "arg.unsupported_value", Category.UNSUPPORTED_ARGUMENT,
                    Severity.ERROR, Status.UNSUPPORTED, usage, index,
                    location=self._loc(kw.value, index),
                    argument=arg, api_name=api_name,
                    message=self._msg("arg.unsupported_value", api=api_name,
                                      argument=arg, value=repr(v), reason=reason),
                    suggestion=suggestion)
            if v in ar.get("semantic_values", []):
                return self._build(
                    "api.semantic_difference", Category.SEMANTIC_DIFFERENCE,
                    Severity.ERROR, Status.SEMANTIC_DIFFERENCE, usage, index,
                    location=self._loc(kw.value, index),
                    argument=arg, api_name=api_name,
                    message=self._msg("api.semantic_difference", api=api_name,
                                      reason=reason),
                    suggestion=suggestion)
        return None

    # ------------------------------------------------------------------ #
    def _check_dtype(self, usage: APIUsage, rule: Optional[dict],
                     index: SourceIndex) -> Optional[Diagnostic]:
        dt = usage.dtype
        per = (rule or {}).get("dtype_rules", {})
        entry = per.get(dt)
        if entry is None and (rule or {}).get("dtype_only"):
            if dt not in rule["dtype_only"]:
                entry = {
                    "status": "unsupported",
                    "reason": ("TensorCore limitation; supported dtypes: "
                               + ", ".join(rule["dtype_only"])),
                    "suggestion": "Use the float version instead.",
                }
        if entry is None:
            entry = self.db.dtype_rules.get(dt)
        if entry is None:
            return None

        status = entry.get("status", "supported")
        if status == "supported":
            return None

        reps = entry.get("replacements", [])
        rep_names = ", ".join(r.get("dtype", "?") for r in reps) if reps else ""
        reason = entry.get("reason", "")
        if self.lang == "zh":
            suggestion = (entry.get("suggestion_zh") or entry.get("suggestion")
                          or entry.get("suggest_zh"))
        else:
            suggestion = entry.get("suggestion")

        if status == "unsupported":
            category, severity, st = (Category.UNSUPPORTED_DTYPE,
                                      Severity.ERROR, Status.UNSUPPORTED)
            mid = "dtype.unsupported"
        elif status == "discouraged":
            category, severity, st = (Category.DISCOURAGED_DTYPE,
                                      Severity.WARNING, Status.DISCOURAGED)
            mid = "dtype.discouraged"
        elif status == "cpu_fallback":
            category, severity, st = (Category.CPU_FALLBACK,
                                      Severity.WARNING, Status.CPU_FALLBACK)
            mid = "dtype.cpu_fallback"
        else:
            return None

        if suggestion is None and rep_names:
            suggestion = ("Consider " + rep_names
                          + " if the value range / semantics permit.")

        d = self._build(
            "dtype." + dt + "." + status, category, severity, st, usage, index,
            dtype=dt,
            message=self._msg(mid, dtype=dt, reason=reason,
                              replacement=rep_names or "a supported dtype"),
            suggestion=suggestion)

        fix_mode = FixMode(entry.get("fix_mode", "manual"))
        d.fix_mode = fix_mode
        if reps:
            d.replacement = Replacement(
                kind="dtype", target=reps[0].get("dtype", ""),
                fix_mode=fix_mode, risk=reps[0].get("risk"))
        if (fix_mode in (FixMode.AUTO_SAFE, FixMode.AUTO_WARNING)
                and reps and usage.dtype_node is not None):
            d.fix = self._dtype_fix(usage.dtype_node, reps[0].get("dtype", ""),
                                    index)
        return d

    def _unknown_dtype_diag(self, usage: APIUsage, index: SourceIndex) -> Diagnostic:
        loc = usage.location
        if usage.dtype_node is not None and hasattr(usage.dtype_node, "lineno"):
            loc = self._loc(usage.dtype_node, index)
        entry = (self.db.message(self.lang, "dtype.unknown")
                 or self.db.message("en", "dtype.unknown") or {})
        d = Diagnostic(
            rule_id="dtype.unknown.dynamic",
            category=Category.UNKNOWN_DYNAMIC,
            severity=Severity.INFO,
            status=Status.UNKNOWN,
            location=loc,
            source_api=(f"{usage.backend}.{usage.operation}"
                        if usage.backend and usage.operation else None),
            message=entry.get("message", "dtype cannot be statically determined."),
            suggestion=entry.get("suggestion"),
            confidence=0.2,
            source_line=index.line_text(loc.line),
        )
        return d

    # ------------------------------------------------------------------ #
    # fix builders
    def _rename_fix(self, func_node, target: str,
                    index: SourceIndex) -> Optional[Fix]:
        if target is None:
            return None
        if isinstance(func_node, ast.Attribute):
            end = index.char_offset(func_node.end_lineno,
                                    func_node.end_col_offset)
            start = end - len(func_node.attr)
        elif isinstance(func_node, ast.Name):
            start, end = index.node_range(func_node)
        else:
            return None
        return Fix(start, end, target, description=f"rename to {target}")

    def _dtype_fix(self, dtype_node, target: str,
                   index: SourceIndex) -> Optional[Fix]:
        if not target:
            return None
        start, end = index.node_range(dtype_node)
        text = index.text[start:end]
        if text and text[0] in "'\"" and text[-1] == text[0]:
            new_text = text[0] + target + text[-1]
        elif "." in text:
            base = text.rsplit(".", 1)[0]
            new_text = base + "." + target
        else:
            new_text = target
        return Fix(start, end, new_text, description=f"dtype -> {target}")

    # ------------------------------------------------------------------ #
    # helpers
    @staticmethod
    def _loc(node, index: SourceIndex) -> SourceLocation:
        return SourceLocation(file=index.filename, line=node.lineno,
                              column=node.col_offset + 1)

    def _suggest(self, rule: dict) -> Optional[str]:
        if self.lang == "zh":
            return rule.get("suggest_zh") or rule.get("suggest")
        return rule.get("suggest")

    def _build(self, rule_id, category, severity, status, usage, index,
               message="", suggestion=None, dtype=None, argument=None,
               api_name=None, location=None) -> Diagnostic:
        loc = location or usage.location
        return Diagnostic(
            rule_id=rule_id,
            category=category,
            severity=severity,
            status=status,
            location=loc,
            source_api=api_name or (f"{usage.backend}.{usage.operation}"
                                    if usage.backend and usage.operation else None),
            dtype=dtype,
            argument=argument,
            message=message,
            suggestion=suggestion,
            source_line=index.line_text(loc.line),
        )

    def _msg(self, message_id: str, **ctx) -> str:
        entry = self.db.message(self.lang, message_id)
        if entry is None:
            entry = self.db.message("en", message_id) or {}
        template = entry.get("message", message_id)
        return template.format_map(_SafeDict(**ctx))


def dedupe(diagnostics: List[Diagnostic]) -> List[Diagnostic]:
    """Keep only primary diagnostics per location (design doc §22)."""
    groups = {}
    for d in diagnostics:
        key = (d.location.file, d.location.line, d.location.column)
        groups.setdefault(key, []).append(d)
    out = []
    for key in sorted(groups):
        g = groups[key]
        maxp = max(PRIORITY[d.category] for d in g)
        out.extend(d for d in g if PRIORITY[d.category] == maxp)
    return out
