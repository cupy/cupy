#!/bin/sh

# non-interactive shell (tmax/nohup) does not load ~/.bashrc
# so need to explicitly source these ~/.bashrc
# run this script from tools/ directory, if numpy-ascend/cupy has been installed
set +u
cd "$(dirname "$0")"

# SCIPY_ARRAY_API must be set BEFORE importing scipy.
# 第一台/第一次：Ascend,  module name still cupy, maybe renamed to numpy_ascend
python scipy_arrayapi_bench.py run \
    --backend ascend \
    --module cupy \
    --size 10000000 \
    --warmup 2 \
    --repeat 10 \
    --output ascend.json

# 第二次：CPU
python scipy_arrayapi_bench.py run \
    --backend cpu \
    --size 10000000 \
    --warmup 2 \
    --repeat 10 \
    --output cpu.json

# 合并
python scipy_arrayapi_bench.py merge \
    --ascend ascend.json \
    --cpu cpu.json \
    --output report.json

# 查看结果
python scipy_arrayapi_bench.py report report.json