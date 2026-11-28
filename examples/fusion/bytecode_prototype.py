"""Validate and benchmark the private bytecode fusion prototype.

Run from a built checkout:
    python -m examples.fusion.bytecode_prototype --json /tmp/bytecode.json

Both CPU submission time and CUDA-event complete-call elapsed time are
reported. Events include submission gaps, not just kernel durations. All
methods allocate outputs normally; validation and compilation are untimed.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time

import numpy as np

import cupy as cp
from cupy._core._fusion_bytecode import fuse
from cupy._core.fusion import Fusion


def score(x):
    energy = (cp.exp(-cp.absolute(x)) * cp.sin(x)**2).sum(axis=1)
    mass = (1 / (1 + cp.exp(-x))).sum(axis=1)
    return cp.sqrt(energy) / (mass + 1)


def helper(x, scale):
    return cp.sin(x) * scale


def nested(x, scale):
    return helper(x, scale) + 1


def allocated_branch(x, flag):
    y = cp.ones(x.shape, dtype=cp.float32)
    if flag:
        return cp.sin(x + y) * cp.cos(x) + x * 2
    return cp.cos(x + y) * cp.sin(x) - x * 2


def measure(function, args, repeat, trials):
    for _ in range(10):
        function(*args)
    cp.cuda.get_current_stream().synchronize()
    samples = []
    for _ in range(trials):
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        before = time.perf_counter_ns()
        for _ in range(repeat):
            function(*args)
        cpu_us = (time.perf_counter_ns() - before) / repeat / 1000
        stop.record()
        stop.synchronize()
        samples.append({'cpu_us': cpu_us,
                        'gpu_elapsed_us': cp.cuda.get_elapsed_time(start, stop)
                        * 1000 / repeat})
    return {key: statistics.median(s[key] for s in samples)
            for key in samples[0]} | {'trials': samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json')
    parser.add_argument('--repeat', type=int, default=100)
    parser.add_argument('--trials', type=int, default=5)
    options = parser.parse_args()
    report = {'cupy': cp.__version__, 'numpy': np.__version__,
              'python': platform.python_version(),
              'commit': subprocess.check_output(
                  ['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'gpu': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'runtime': cp.cuda.runtime.runtimeGetVersion(),
              'repeat': options.repeat, 'trials': options.trials, 'rows': []}
    with cp.cuda.Stream(non_blocking=True):
        for shape in ((64, 64), (4096, 64), (32768, 512)):
            x = cp.linspace(-1, 1, np.prod(shape),
                            dtype=cp.float32).reshape(shape)
            for function, args in ((score, (x,)),
                                   (nested, (x, np.float32(0.75))),
                                   (allocated_branch, (x, True))):
                expected = function(*args)
                methods = [('eager', function), ('bytecode', fuse(function))]
                if function is not allocated_branch:
                    methods.insert(1, ('legacy', Fusion(function)))
                for label, call in methods:
                    before = time.perf_counter()
                    actual = call(*args)
                    cp.cuda.get_current_stream().synchronize()
                    first_call_seconds = time.perf_counter() - before
                    assert actual.dtype == expected.dtype
                    cp.testing.assert_allclose(actual, expected,
                                               rtol=3e-5, atol=2e-6)
                    row = {'workload': function.__name__, 'shape': list(shape),
                           'method': label,
                           'first_call_seconds': first_call_seconds,
                           **measure(call, args, options.repeat,
                                     options.trials)}
                    if label == 'bytecode':
                        row['diagnostics'] = call.diagnostics
                    report['rows'].append(row)
                    print(function.__name__, shape, label,
                          f'CPU {row["cpu_us"]:.2f} us, '
                          f'GPU elapsed {row["gpu_elapsed_us"]:.2f} us',
                          flush=True)
    if options.json:
        with open(options.json, 'w') as output:
            json.dump(report, output, indent=2)
            output.write('\n')


if __name__ == '__main__':
    main()
