# Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
from pathlib import Path

import cupy as cp


def _get_numSM():

    device_id = cp.cuda.Device()

    return device_id.attributes["MultiProcessorCount"]


def _get_max_smem():

    device_id = cp.cuda.Device()

    return device_id.attributes["MaxSharedMemoryPerBlock"]


def _get_max_tpb():

    device_id = cp.cuda.Device()

    return device_id.attributes["MaxThreadsPerBlock"]


def _get_max_gdx():

    device_id = cp.cuda.Device()

    return device_id.attributes["MaxGridDimX"]


def _get_max_gdy():

    device_id = cp.cuda.Device()

    return device_id.attributes["MaxGridDimY"]


def _get_tpb_bpg():

    numSM = _get_numSM()
    threadsperblock = 512
    blockspergrid = numSM * 20

    return threadsperblock, blockspergrid


def _get_function(fatbin, func):

    dir = os.path.dirname(Path(__file__).parent)

    module = cp.RawModule(
        path=dir + fatbin,
    )
    return module.get_function(func)


def _print_atts(func):
    if os.environ.get("CUSIGNAL_DEV_DEBUG") == "True":
        print("name:", func.kernel.name)
        print("max_threads_per_block:", func.kernel.max_threads_per_block)
        print("num_regs:", func.kernel.num_regs)
        print(
            "max_dynamic_shared_size_bytes:",
            func.kernel.max_dynamic_shared_size_bytes,
        )
        print("shared_size_bytes:", func.kernel.shared_size_bytes)
        print(
            "preferred_shared_memory_carveout:",
            func.kernel.preferred_shared_memory_carveout,
        )
        print("const_size_bytes:", func.kernel.const_size_bytes)
        print("local_size_bytes:", func.kernel.local_size_bytes)
        print("ptx_version:", func.kernel.ptx_version)
        print("binary_version:", func.kernel.binary_version)
        print()
