#!/usr/bin/env bash
# Run every test file of one directory in its own pytest invocation, so a
# stalled / crashed / aborted test cannot take down the rest of the suite.
#
# Usage: run_pytest_dir.sh <test-dir> <log-tag> [timeout-seconds]
#
# Logs:  tests/logs/20261010-logs/<tag>/<test file with / -> __>.log
#        tests/logs/20261010-logs/<tag>/summary.txt
set -u

REPO=/root/numpy-ascend
TESTDIR=${1:?test dir required}
TAG=${2:?log tag required}
TIMEOUT=${3:-600}
LOGDIR=$REPO/tests/logs/20261010-logs/$TAG
mkdir -p "$LOGDIR"

source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
# pytest's filterwarnings in pyproject.toml resolves cupy.exceptions.* at
# startup, so the repo root must be importable.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO" || exit 1

# Files tests/conftest.py ignores during directory collection (CUDA-only:
# nvrtc/cufft/cublas/curand/cusparse/_cub_reduction/_scan_for_test). pytest
# does NOT consult pytest_ignore_collect for paths given explicitly on the
# command line, so per-file invocations must skip them here.
SKIP_RE='(test_cublas|test_core|test_cub_reduction|test_function|test_ndarray_reduction|test_raw|test_scan|test_cache|test_callback|test_sumprod|test_search)\.py$'

: > "$LOGDIR/summary.txt"
find "$TESTDIR" -maxdepth 1 -name 'test_*.py' | sort |
while read -r f; do
    rel=${f#tests/}
    if echo "$rel" | grep -qE "$SKIP_RE"; then
        printf '%-52s %-16s\n' "$rel" "SKIPPED(cuda-only)" |
            tee -a "$LOGDIR/summary.txt"
        continue
    fi
    name=$(echo "$rel" | sed 's|/|__|g; s|\.py$||')
    start=$(date +%s)
    timeout "$TIMEOUT" python -m pytest "$f" -q --tb=short -rf \
        > "$LOGDIR/$name.log" 2>&1
    rc=$?
    end=$(date +%s)
    dur=$((end - start))
    last=$(tail -1 "$LOGDIR/$name.log" | cut -c1-90)
    case $rc in
        0)   status=PASSED ;;
        124) status=STALLED ;;
        139) status=SEGFAULT ;;
        134) status=ABORTED ;;
        *)   status="FAILED(rc=$rc)" ;;
    esac
    printf '%-52s %-16s %5ss  %s\n' "$rel" "$status" "$dur" "$last" |
        tee -a "$LOGDIR/summary.txt"
done
echo "=== $TAG DONE ===" | tee -a "$LOGDIR/summary.txt"
