#!/bin/bash

set -uex

# Note: Linux CI image uses relatime.

# Always remove cuFFT static library cache; .so files are sufficient at runtime.
echo "Cleaning up cuFFT static library cache:"
find "${CUPY_CACHE_DIR}" -type f \
    -name "libcufft_static_*.a" \
    -print -delete | wc -l

# Expire misc cache in 30 days.
echo "Cleaning up misc cache:"
find "${CUPY_CACHE_DIR}" -type f \( \
    -name "jitify_*.json" -or \
    -name "cupy_callback_*.so" \
    \) -atime +30 -mtime +30 -ctime +30 -print -delete | wc -l

# Expire *.cubin cache in 30 days, and also limit by total size.
python3 .pfnci/trim_cupy_kernel_cache.py --max-size $((3*1024*1024*1024)) --expiry $((30*24*60*60)) --rm

# Limit ccache by total size.
ccache --max-size 0.5Gi --cleanup --show-stats
