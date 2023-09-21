import gc
import math
import os

import numpy, cupy
from cupy.cuda import nccl
from cupyx.profiler import benchmark
from cupyx.distributed._array import distributed_array
from cupy.testing import assert_array_equal
from tests.cupyx_tests.distributed_tests.test_linalg import ArrayConfig
from tests.cupyx_tests.distributed_tests.test_linalg import MatMulConfig
from tests.cupyx_tests.distributed_tests.test_linalg import make_2d_config


cuda_visible_devices = os.environ.get('BENCHMARK_DEVICES', '0,1,2,3')
devices = [int(dev) for dev in cuda_visible_devices.split(',')]


def assign_devices(xs):
    res = []
    for x in xs:
        if isinstance(x, dict):
            x = {devices[dev]: val for dev, val in x.items()}
        elif isinstance(x, MatMulConfig):
            x.a.index_map = {devices[dev]: val
                             for dev, val in x.a.index_map.items()}
            x.b.index_map = {devices[dev]: val
                             for dev, val in x.b.index_map.items()}
        res.append(x)

    return res


def repeat(f, n_dev):
    if n_dev != 4:
        return
    for dev in devices:
        with cupy.cuda.Device(dev):
            cupy.cuda.get_current_stream().synchronize()
    from time import sleep
    for i in range(5):
        sleep(0.2)
        f()
        sleep(0.2)
        print(f'{i+1}/5')
        gc.collect()


def bench(f, n_dev=4):
    if 'PROFILING' in os.environ.keys():
        return repeat(f, n_dev)

    for dev in devices:
        with cupy.cuda.Device(dev):
            cupy.cuda.get_current_stream().synchronize()
    print(benchmark(f, n_repeat=25, devices=tuple(devices[:n_dev])))


def make_comms():
    if not nccl.available:
        return None
    comms_list = nccl.NcclCommunicator.initAll(4)
    return {dev: comm for dev, comm in zip(range(4), comms_list)}


comms = make_comms()


streams = {}


for dev in devices:
    with cupy.cuda.Device(dev):
        streams[dev] = cupy.cuda.Stream()
        streams[dev].__enter__()


cupy.cuda.Device(devices[0]).use()
