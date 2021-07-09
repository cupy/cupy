export CACHE_DIR="${CACHE_DIR:-/tmp}"
export CCACHE_DIR="${CACHE_DIR}/.ccache"
export CCACHE_NOHASHDIR="true"
export CUDA_CACHE_PATH="${CACHE_DIR}/.nv"
export CUPY_CACHE_DIR="${CACHE_DIR}/.cupy/kernel_cache"

export CUPY_TEST_GPU_LIMIT="${GPU:--1}"
export CUPY_TEST_FULL_COMBINATION="0"
export CUPY_DUMP_CUDA_SOURCE_ON_ERROR="1"

# For compatibility with Jenkins with docker wrapper
if [[ "${UID}" != "0" && "${HOME:-/}" == "/" ]]; then
    export HOME=/tmp
fi

# Show GPU statistics
( which nvidia-smi &> /dev/null ) && nvidia-smi
( which hipconfig &> /dev/null ) && hipconfig

# Show environment variables
env
