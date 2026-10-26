#!/usr/bin/env bash
# Run every test suite except math_tests (already covered) and the CUDA-only
# tests, one test file per pytest invocation. See run_pytest_dir.sh.
set -u
REPO=/root/numpy-ascend
cd "$REPO" || exit 1

DIRS=$(find tests/cupy_tests tests/cupyx_tests tests/import_tests tests/typing_tests \
        -type d \
        ! -path '*__pycache__*' \
        ! -path '*/cuda_tests*' \
        ! -path '*/math_tests*' |
      sort)

for d in $DIRS; do
    tag=$(echo "$d" | sed 's|^tests/||; s|/|_|g')
    if [ -z "$(find "$d" -maxdepth 1 -name 'test_*.py' -print -quit)" ]; then
        continue
    fi
    # Resume support: skip a suite whose per-file logs are already complete.
    if grep -q "DONE ===" "$REPO/tests/logs/20261010-logs/$tag/summary.txt" 2>/dev/null; then
        echo "### $d (already done, skipping)"
        continue
    fi
    echo "### $d"
    bash "$REPO/tests/logs/run_pytest_dir.sh" "$d" "$tag" 600
done
echo "ALL SUITES DONE"
