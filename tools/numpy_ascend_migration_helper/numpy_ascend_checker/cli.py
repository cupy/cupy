"""numpy-ascend-check CLI.

Dry-run analysis is the default; `--fix` performs source rewrite (only
AUTO_SAFE replacements unless --risky), `--fix --dry-run` prints a diff.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import sys
from pathlib import Path
from typing import List, Optional

from .analyzer import analyze_tree
from .engine import RuleEngine, dedupe
from .fixer import apply_fixes
from .models import Diagnostic, Severity
from .reporter import JsonReporter, TextReporter, summarize
from .resolver import build_symbol_table
from .rule_db import RuleDB
from .source import SourceIndex

__version__ = "0.1.0"

DEFAULT_EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", "build",
                        "dist", ".codebuddy", ".eggs", "node_modules"}

RULES_DIR = Path(__file__).resolve().parent / "rules"


def collect_files(paths: List[str], include=None, exclude=None,
                  exclude_dirs=None) -> List[Path]:
    include = include or []
    exclude = exclude or []
    exclude_dirs = set(exclude_dirs or []) | DEFAULT_EXCLUDE_DIRS
    files: List[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for f in sorted(pp.rglob("*.py")):
                if any(part in exclude_dirs for part in f.parts):
                    continue
                rel = f.as_posix()
                if exclude and any(
                        fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(f.name, pat)
                        for pat in exclude):
                    continue
                if include and not any(
                        fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(f.name, pat)
                        for pat in include):
                    continue
                files.append(f)
        elif pp.is_file():
            files.append(pp)
        else:
            print(f"warning: path not found: {p}", file=sys.stderr)
    # dedupe, keep order
    seen = set()
    unique = []
    for f in files:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            unique.append(f)
    return unique


def analyze_file(path: Path, engine: RuleEngine):
    """Returns (index, diagnostics, usage_count) or None on parse failure."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"warning: cannot read {path}: {e}", file=sys.stderr)
        return None
    index = SourceIndex(text, str(path))
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as e:
        print(f"warning: syntax error in {path}: {e}", file=sys.stderr)
        return None
    st = build_symbol_table(tree)
    usages, dtype_refs = analyze_tree(tree, st, index)
    diagnostics: List[Diagnostic] = []
    for usage in usages:
        diagnostics.extend(engine.analyze_usage(usage, index))
    for node, backend in dtype_refs:
        d = engine.analyze_dtype_ref(node, backend, index)
        if d:
            diagnostics.append(d)
    return index, dedupe(diagnostics), len(usages)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="numpy-ascend-check",
        description="NumPy/CuPy -> Ascend (numpy-ascend) compatibility "
                    "migration analyzer",
    )
    parser.add_argument("paths", nargs="+",
                        help="python file(s) or project directories")
    parser.add_argument("--fix", action="store_true",
                        help="apply AUTO_SAFE source rewrites (default: "
                             "analyze only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --fix: show diff without writing")
    parser.add_argument("--risky", action="store_true",
                        help="with --fix: also apply AUTO_WARNING fixes "
                             "(e.g. float64 -> float32)")
    parser.add_argument("--backup", action="store_true",
                        help="write <file>.bak before rewriting")
    parser.add_argument("--format", choices=("text", "json"),
                        default="text", dest="format")
    parser.add_argument("--output", default=None,
                        help="write report to file instead of stdout")
    parser.add_argument("--lang", choices=("en", "zh"), default="en",
                        help="message language (default: en)")
    parser.add_argument("--fail-on-error", action="store_true",
                        help="exit with code 1 if any ERROR was reported "
                             "(for CI)")
    parser.add_argument("--report-unknown-api", action="store_true",
                        help="report APIs missing from the rule DB as "
                             "UNKNOWN (INFO)")
    parser.add_argument("--include", action="append", default=[],
                        metavar="GLOB",
                        help="only scan files matching GLOB (repeatable)")
    parser.add_argument("--exclude", action="append", default=[],
                        metavar="GLOB",
                        help="skip files matching GLOB (repeatable)")
    parser.add_argument("--exclude-dir", action="append", default=[],
                        metavar="NAME",
                        help="skip directories with this name (repeatable)")
    parser.add_argument("--rules-dir", default=None,
                        help="alternate rule DB directory")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    db = RuleDB(args.rules_dir or RULES_DIR)
    engine = RuleEngine(db, lang=args.lang,
                        report_unknown_api=args.report_unknown_api)
    files = collect_files(args.paths, include=args.include,
                          exclude=args.exclude,
                          exclude_dirs=args.exclude_dir)

    all_diags: List[Diagnostic] = []
    total_usages = 0
    parse_failures = 0

    for f in files:
        result = analyze_file(f, engine)
        if result is None:
            parse_failures += 1
            continue
        _, diags, nusages = result
        all_diags.extend(diags)
        total_usages += nusages

        if args.fix:
            applied, skipped, diff = apply_fixes(
                f, diags, risky=args.risky, dry_run=args.dry_run,
                backup=args.backup)
            if applied or skipped:
                mode = "would apply" if args.dry_run else "applied"
                print(f"{f}: {mode} {applied} fix(es)"
                      + (f", skipped {skipped} overlapping" if skipped else ""))
            if diff:
                print(diff)

    summary = summarize(all_diags, files=len(files), usages=total_usages,
                        parse_failures=parse_failures)

    if args.format == "json":
        report = JsonReporter().render(all_diags, summary, db)
        text = json.dumps(report, indent=2, ensure_ascii=False)
    else:
        text = TextReporter().render(all_diags, summary)

    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"report written to {args.output}")
    else:
        print(text)

    if args.fail_on_error and summary["errors"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
