#!/usr/bin/env bash
# Run each math_tests test file in its own pytest invocation so a stalled /
# crashing test cannot take the whole suite down, and write one log per file
# under tests/logs/<test-date>-logs/.
set -u

REPO=/root/numpy-ascend
MATH=$REPO/tests/cupy_tests/math_tests
LOGDIR=$REPO/tests/logs/20261010-logs
TIMEOUT=${TIMEOUT:-600}
mkdir -p "$LOGDIR"

source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
# pytest's filterwarnings in pyproject.toml resolves cupy.exceptions.* at
# startup, so the repo root must be importable even though we cd into
# tests/cupy_tests/math_tests.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$MATH" || exit 1

: > "$LOGDIR/summary.txt"
# Files tests/conftest.py ignores during directory collection. pytest does NOT
# consult pytest_ignore_collect for paths given explicitly on the command line,
# so per-file invocations must skip them here.
SKIP_FILES="test_sumprod.py"
for f in test_*.py; do
    name="${f%.py}"
    case " $SKIP_FILES " in
        *" $f "*)
            printf '%-28s %-14s\n' "$name" "SKIPPED(cuda-only)" \
                | tee -a "$LOGDIR/summary.txt"
            continue ;;
    esac
    start=$(date +%s)
    timeout "$TIMEOUT" python -m pytest "$f" -q --tb=short -rf \
        > "$LOGDIR/$name.log" 2>&1
    rc=$?
    end=$(date +%s)
    dur=$((end - start))
    last=$(tail -1 "$LOGDIR/$name.log" | cut -c1-100)
    case $rc in
        0)   status=PASSED ;;
        124) status=STALLED ;;
        139) status=SEGFAULT ;;
        134) status=ABORTED ;;
        *)   status="FAILED(rc=$rc)" ;;
    esac
    printf '%-28s %-14s %5ss  %s\n' "$name" "$status" "$dur" "$last" \
        | tee -a "$LOGDIR/summary.txt"
done
echo "ALL DONE" | tee -a "$LOGDIR/summary.txt"
