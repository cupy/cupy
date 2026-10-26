#!/usr/bin/env python
"""Aggregate the per-file pytest logs written by run_math_tests.sh /
run_pytest_dir.sh into one pass-rate table.

Only each log's *final* pytest summary line is counted, so numbers that merely
appear inside assertion messages (e.g. CANN's "open device 139 failed") do not
pollute the totals.

Usage: python logs/aggregate_results.py [logs/20261010-logs]
"""
import glob
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'tests/logs/20261010-logs'

# A pytest terminal summary line, e.g.
#   "66 failed, 142 passed, 7 skipped in 45.20s"
#   "2006 failed, 640 passed, 54 errors in 147.76s (0:02:27)"
SUMMARY_LINE = re.compile(
    r'(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)\b.*in [\d.]+s')
COUNT = re.compile(r'(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)\b')

COLS = ('passed', 'failed', 'error', 'skipped', 'xfailed')


def parse(path):
    """Return the counts of the last summary line in `path` ({} if none)."""
    counts = {}
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if not SUMMARY_LINE.search(line):
                    continue
                counts = {}
                for n, kind in COUNT.findall(line):
                    kind = 'error' if kind.startswith('error') else kind
                    counts[kind] = counts.get(kind, 0) + int(n)
    except OSError as e:
        print('cannot read %s: %s' % (path, e))
    return counts


def main():
    logs = sorted(glob.glob(os.path.join(ROOT, '**', '*.log'), recursive=True))
    total = {c: 0 for c in COLS}
    files = 0
    rows = []
    for log in logs:
        counts = parse(log)
        if not counts:
            continue
        files += 1
        f, p = counts.get('failed', 0), counts.get('passed', 0)
        rows.append((f, p, os.path.relpath(log, ROOT)))
        for c in COLS:
            total[c] += counts.get(c, 0)
    rows.sort(reverse=True)
    for f, p, log in rows[:15]:
        print('%7d failed %7d passed  %s' % (f, p, log))
    print('--- %d result files ---' % files)
    print({c: n for c, n in total.items() if n})
    ran = total['passed'] + total['failed']
    if ran:
        print('pass rate: %.1f%% (%d ran)'
              % (100.0 * total['passed'] / ran, ran))


if __name__ == '__main__':
    main()
