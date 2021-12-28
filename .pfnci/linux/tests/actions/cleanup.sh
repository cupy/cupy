#!/bin/bash

set -uex

python3 .pfnci/trim_cupy_kernel_cache.py --max-size $((3*1024*1024*1024)) --rm
ccache --max-size 0.5Gi --cleanup --show-stats
