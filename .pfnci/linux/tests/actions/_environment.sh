export CACHE_DIR="${CACHE_DIR:-/tmp}"
export CCACHE_DIR="${CACHE_DIR}/.ccache"
export CCACHE_NOHASHDIR="true"
export CUDA_CACHE_PATH="${CACHE_DIR}/.nv"
export CUPY_CACHE_DIR="${CACHE_DIR}/.cupy/kernel_cache"

export CUPY_TEST_GPU_LIMIT="${GPU:--1}"
export CUPY_DUMP_CUDA_SOURCE_ON_ERROR="1"

if [[ "${PULL_REQUEST:-}" == "" ]]; then
    # When testing branches, test full matrix to generate caches for all combinations.
    # Note: this may be overridden by individual test scripts.
    export CUPY_TEST_FULL_COMBINATION="1"
else
    # When testing pull-requests, make test combinations sparse.
    export CUPY_TEST_FULL_COMBINATION="0"
fi

# For compatibility with Jenkins with docker wrapper
if [[ "${UID}" != "0" && "${HOME:-/}" == "/" ]]; then
    export HOME=/tmp
fi

# Add PATH for commands installed via `pip install --user`
export PATH="${HOME}/.local/bin:${PATH}"

# Show GPU statistics
( which nvidia-smi &> /dev/null ) && nvidia-smi
( which hipconfig &> /dev/null ) && hipconfig

# Show environment variables
env

# Switch to a temporary directory to run tests
if ! touch .; then
    _src_dir="$(mktemp -d)"
    echo "Source directory ($(pwd)) is read-only; copying the source tree to ${_src_dir} and changing the current directory"
    cp -a . "${_src_dir}"
    pushd "${_src_dir}"
    export CCACHE_BASEDIR="${_src_dir}"
fi
