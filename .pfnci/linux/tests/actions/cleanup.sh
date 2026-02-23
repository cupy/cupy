#!/bin/bash

set -uex

# Note: Linux CI image uses relatime.

# Always remove cuFFT static library and FFT callback caches.
# .a files are only used to generate .so files and are not needed at runtime.
# FFT callback .so files are for legacy callbacks (to be removed) and are too
# large to cache; regenerating them every run only takes a few minutes.
echo "Cleaning up cuFFT cache:"
find "${CUPY_CACHE_DIR}" -type f \( \
    -name "libcufft_static_*.a" -or \
    -name "cupy_callback_*.so" \
    \) -print -delete | wc -l

# Expire misc cache in 30 days.
echo "Cleaning up misc cache:"
find "${CUPY_CACHE_DIR}" -type f \
    -name "jitify_*.json" \
    -atime +30 -mtime +30 -ctime +30 -print -delete | wc -l

# Purge all local *.cubin cache (they are stored separately via GCS kernel cache backend)
python3 .pfnci/trim_cupy_kernel_cache.py --max-size 0 --rm

# Limit ccache by total size.
ccache --max-size 0.5Gi --cleanup --show-stats
