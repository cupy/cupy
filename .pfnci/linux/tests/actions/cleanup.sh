#!/bin/bash

set -uex

python3 .pfnci/trim_cupy_kernel_cache.py --max-size $((5*1024*1024*1024)) --rm
