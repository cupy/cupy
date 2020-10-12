#!/bin/bash
# run.sh is a script to run unit tests inside Docker.  This should be injected
# by and called from script.sh.
#
# Usage: run.sh [target]
# - target is a test target (e.g., "py37").
#
# Environment variables:
# - GPU (default: 0) ... Set a number of GPUs to GPU.
#       CAVEAT: Setting GPU>0 disables non-GPU tests, and setting GPU=0 disables
#               GPU tests.
# - SPREADSHEET_ID ... Set SPREADSHEET_ID (e.g.,
#       "1u5OYiPOL3XRppn73XBSgR-XyDuHKb_4Ilmx1kgJfa-k") to enable xpytest to
#       report to a spreadsheet.

set -eux

export CUPY_CI=flexci

cp -a /src /cupy
cp /cupy/setup.cfg /
cd /

# Remove pyc files.  When the CI is triggered with a user's local files, pyc
# files generated on the user's local machine and they often cause failures.
find /cupy -name "*.pyc" -exec rm -f {} \;

TARGET="$1"
: "${GPU:=0}"
: "${XPYTEST_NUM_THREADS:=$(nproc)}"

# Use multi-process service to prevent GPU flakiness caused by running many
# processes on a GPU.  Specifically, it seems running more than 16 processes
# sometimes causes "cudaErrorLaunchFailure: unspecified launch failure".
if (( GPU > 0 )); then
    nvidia-smi -c EXCLUSIVE_PROCESS
    nvidia-cuda-mps-control -d
fi

################################################################################
# Test functions
################################################################################

# test_py37 is a test function for cupy.py37.
test_py37() {
  #-----------------------------------------------------------------------------
  # Configure parameters
  #-----------------------------------------------------------------------------
  export CUPY_TEST_GPU_LIMIT="${GPU}"
  marker='not slow'
  bucket="${GPU}"

  #-----------------------------------------------------------------------------
  # Install CuPy
  #-----------------------------------------------------------------------------
  CUPY_NVCC_GENERATE_CODE=arch=compute_70,code=sm_70 \
      python3.7 -m pip install /cupy[test] 2>&1 >/tmp/install.log &
  install_pid=$!

  if ! wait $install_pid; then
    cat /tmp/install.log
    exit 1
  fi

  #-----------------------------------------------------------------------------
  # CuPy tests
  #-----------------------------------------------------------------------------
  xpytest_args=(
      --python=python3.7 -m "${marker}"
      --bucket="${bucket}" --thread="$(( XPYTEST_NUM_THREADS / bucket ))"
      --hint="/cupy/.pfnci/hint.pbtxt"
      --retry=1
  )
  if [ "${SPREADSHEET_ID:-}" != '' ]; then
    xpytest_args+=(--spreadsheet_id="${SPREADSHEET_ID}")
  fi
  # TODO(niboshi): Allow option pass-through (https://github.com/chainer/xpytest/issues/14)
  export PYTEST_ADDOPTS=-rfEX
  # TODO(imos): Enable xpytest to support python_files setting in setup.cfg.
  OMP_NUM_THREADS=1 xpytest "${xpytest_args[@]}" \
      '/cupy/tests/**/test_*.py' \
      && :
  py_test_status=$?

  #-----------------------------------------------------------------------------
  # Finalize
  #-----------------------------------------------------------------------------
  echo "py_test_status=${py_test_status}"
  exit ${py_test_status}
}

# test_py27and35 is a test function for cupy.py27and35.
# Despite the name, Python 2.7 is no longer tested in the master branch.
# TODO(niboshi): Completely remove Python 2.7 after discontinuing v6 series.
test_py27and35() {
  #-----------------------------------------------------------------------------
  # Configure parameters
  #-----------------------------------------------------------------------------
  export CUPY_TEST_GPU_LIMIT="${GPU}"
  marker='not slow'
  bucket="${GPU}"

  #-----------------------------------------------------------------------------
  # Install CuPy
  #-----------------------------------------------------------------------------
  # Install CuPy for python3.5.
  CUPY_NVCC_GENERATE_CODE=arch=compute_70,code=sm_70 \
      python3.5 -m pip install /cupy[test] 2>&1 >/tmp/install-py35.log &
  install_pid=$!

  if ! wait $install_pid; then
    cat /tmp/install-py35.log
    exit 1
  fi

  #-----------------------------------------------------------------------------
  # Test python3.5
  #-----------------------------------------------------------------------------
  xpytest_args=(
      --python=python3.5 -m "${marker}"
      --bucket="${bucket}" --thread="$(( XPYTEST_NUM_THREADS / bucket ))"
      --hint="/cupy/.pfnci/hint.pbtxt"
      --retry=1
  )
  if [ "${SPREADSHEET_ID:-}" != '' ]; then
    xpytest_args+=(--spreadsheet_id="${SPREADSHEET_ID}")
  fi
  # TODO(niboshi): Allow option pass-through (https://github.com/chainer/xpytest/issues/14)
  export PYTEST_ADDOPTS=-rfEX
  # NOTE: PYTHONHASHSEED=0 is necessary to use pytest-xdist.
  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 xpytest "${xpytest_args[@]}" \
      '/cupy/tests/**/test_*.py' && :
  py35_test_status=$?

  #-----------------------------------------------------------------------------
  # Finalize
  #-----------------------------------------------------------------------------
  echo "py35_test_status=${py35_test_status}"
  exit ${py35_test_status}
}


################################################################################
# Bootstrap
################################################################################
case "${TARGET}" in
  'py37' ) test_py37;;
  'py27and35' ) test_py27and35;;
  * ) echo "Unsupported target: ${TARGET}" >&2; exit 1;;
esac
