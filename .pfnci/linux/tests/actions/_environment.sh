# Automatically use "/tmp" as cache directory if not set
_cache_dir="${CACHE_DIR:-/tmp}"

# Setup ccache env vars
export CCACHE_DIR="${_cache_dir}/.ccache"
export CCACHE_NOHASHDIR="true"

# Setup CUDA env vars
export CUDA_CACHE_PATH="${_cache_dir}/.nv"

# Setup CuPy env vars (for build-time)
export CUPY_NUM_NVCC_THREADS=5
export CUPY_NUM_BUILD_JOBS=8

# Setup CuPy env vars (for runtime)
export CUPY_CACHE_DIR="${_cache_dir}/.cupy/kernel_cache"
export CUPY_DUMP_CUDA_SOURCE_ON_ERROR="1"
export CUPY_NVRTC_USE_PCH="1"

# Setup CuPy env vars (for unit tests)
export CUPY_TEST_GPU_LIMIT="${GPU:--1}"
if [[ "${PULL_REQUEST:-}" == "" ]]; then
    # When testing branches, test full matrix to generate caches for all combinations.
    # Note: this may be overridden by individual test scripts.
    export CUPY_TEST_FULL_COMBINATION="1"
else
    # When testing pull-requests, make test combinations sparse.
    export CUPY_TEST_FULL_COMBINATION="0"
fi

# Add PATH for commands installed via `pip install --user`
export PATH="${HOME}/.local/bin:${PATH}"

# Switch to a temporary directory to run tests
if [[ ${SHELL_MODE:-no} != yes ]]; then
    _src_dir="$(mktemp -d)"
    echo "Copying the source tree from $(pwd) to ${_src_dir} and changing the current directory"
    cp -a . "${_src_dir}"
    pushd "${_src_dir}"
    export CCACHE_BASEDIR="${_src_dir}"
fi
