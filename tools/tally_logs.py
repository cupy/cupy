#!/usr/bin/env python3
"""Tally pytest logs under tests/logs/20261011-logs and dump failures."""
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, "tests", "logs", "20261011-logs")

KEYWORD = re.compile(r"(\d+) (passed|failed|error(?:s|ed)?|skipped|xfailed|xpassed)")

def parse_summary(tail: str):
    """Parse the LAST pytest summary line, e.g. '154 failed, 8633 passed,
    7 skipped, 21 xfailed in 42.50s'. Returns dict or None."""
    for line in reversed(tail.rstrip().splitlines()):
        if " in " not in line:
            continue
        if not re.search(r"\d+ (passed|failed)", line):
            continue
        counts = {w: n for n, w in KEYWORD.findall(line)}
        if not counts:
            continue
        out = {"passed": 0, "failed": 0, "error": 0, "skipped": 0,
               "xfailed": 0, "xpassed": 0}
        for k, v in counts.items():
            k2 = "error" if k.startswith("error") else k
            out[k2] += int(v)
        return out
    return None
FAILED_LINE = re.compile(r"^(?:FAILED|ERROR) (tests/\S+)", re.MULTILINE)

tally = defaultdict(lambda: dict(passed=0, failed=0, error=0, skipped=0,
                                 xfailed=0, xpassed=0, nlogs=0, stalled=0))
failures = defaultdict(list)  # test id -> [dirs]
errors = defaultdict(list)    # test id -> [dirs]

for dirpath, _, files in os.walk(LOGDIR):
    for fn in sorted(files):
        if not fn.endswith(".log"):
            continue
        rel = os.path.relpath(dirpath, LOGDIR)
        top = rel.split(os.sep)[0] if rel != "." else "(root)"
        path = os.path.join(dirpath, fn)
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        tail = text[-4000:]
        counts = parse_summary(tail)
        if counts is None:
            if "STALLED" in tail or "Timeout" in tail or "timeout" in tail:
                tally[top]["stalled"] += 1
            continue
        t = tally[top]
        t["nlogs"] += 1
        for k, v in counts.items():
            t[k] += v
        # collect FAILED/ERROR test ids (last 15000 chars is enough)
        for lm in FAILED_LINE.finditer(text[-15000:]):
            tid = lm.group(1)
            if lm.group(0).startswith("ERROR"):
                errors[tid].append(f"{top}/{fn}")
            else:
                failures[tid].append(f"{top}/{fn}")

print("== per-directory tally ==")
grand = dict(passed=0, failed=0, error=0, skipped=0, xfailed=0, xpassed=0)
for top in sorted(tally):
    t = tally[top]
    print(f"{top}: logs={t['nlogs']} passed={t['passed']} failed={t['failed']} "
          f"error={t['error']} skipped={t['skipped']} xfailed={t['xfailed']} "
          f"stalled_logs={t['stalled']}")
    for k in grand:
        grand[k] += t[k]
print(f"TOTAL: {grand}")

print("\n== distinct FAILED test ids:", len(failures), "==")
for tid in sorted(failures):
    print(tid)

print("\n== distinct ERROR test ids:", len(errors), "==")
for tid in sorted(errors):
    print(tid, "|", ",".join(sorted(set(errors[tid]))[:2]))
