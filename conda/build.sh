#!/usr/bin/env bash
# Build script for the numpy-ascend-cann90 conda package.
set -euo pipefail

# Select the Ascend backend at compile time (see
# install/cupy_builder/features/ascend.py).
export CUPY_INSTALL_USE_ASCEND=1

# cann-toolkit's activate.d scripts normally export ASCEND_HOME_PATH;
# fall back to the newest cann-* tree under $CONDA_PREFIX when they don't
# (e.g. older toolkit packages without activation hooks).
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    _cann_dir="$(ls -d "${CONDA_PREFIX}"/Ascend/cann-* 2>/dev/null | tail -n 1 || true)"
    if [ -z "${_cann_dir}" ]; then
        echo "ERROR: ASCEND_HOME_PATH is not set and no Ascend/cann-* found under ${CONDA_PREFIX}" >&2
        exit 1
    fi
    export ASCEND_HOME_PATH="${_cann_dir}"
fi
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
test -f "${ASCEND_HOME_PATH}/version.cfg" \
    || test -f "${ASCEND_HOME_PATH}/compiler/version.info" \
    || test -f "${ASCEND_HOME_PATH}/opp/version.info" \
    || { echo "ERROR: ${ASCEND_HOME_PATH} does not look like a CANN install" >&2; exit 1; }

# --no-build-isolation: host env already provides setuptools/cython/numpy
# pinned in meta.yaml; --no-deps: run deps declared in meta.yaml.
"${PYTHON}" -m pip install . --no-deps --no-build-isolation -vv
