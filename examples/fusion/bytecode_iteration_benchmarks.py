"""Compare scheduled kernels with a forced one-block control and score().

Run from the built checkout:
    python -m examples.fusion.bytecode_iteration_benchmarks --json /tmp/r.json

Optional --baseline-frontend loads a saved _fusion_bytecode.py to compare warm
score calls before/after scheduler changes in the same process. Timings include
normal output/workspace allocation; compilation and validation are untimed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import cupy as cp

from cupy._core._fusion_bytecode import fuse
from cupy._core._fusion_bytecode_kernel import Executable
from examples.fusion.bytecode_iteration import filter_agg, sigma_clipped_mean
from examples.fusion.bytecode_prototype import score


def measure(methods, args, repeat, trials):
    samples = {name: [] for name in methods}
    for function in methods.values():
        for _ in range(5):
            function(*args)
    cp.cuda.get_current_stream().synchronize()
    for trial in range(trials):
        order = list(methods)
        if trial % 2:
            order.reverse()
        for name in order:
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            before = time.perf_counter_ns()
            for _ in range(repeat):
                methods[name](*args)
            cpu = (time.perf_counter_ns() - before) / repeat / 1000
            stop.record()
            stop.synchronize()
            samples[name].append({
                'cpu_us': cpu,
                'elapsed_us': cp.cuda.get_elapsed_time(start, stop)
                * 1000 / repeat})
    return {name: {'trials': values, **{
        key: statistics.median(row[key] for row in values)
        for key in ('cpu_us', 'elapsed_us')}}
        for name, values in samples.items()}


def baseline_fuse(path):
    name = 'cupy._core._fusion_before_scheduling'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.fuse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', required=True)
    parser.add_argument('--baseline-frontend', type=Path)
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--trials', type=int, default=5)
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report = {'python': platform.python_version(), 'cupy': cp.__version__,
              'gpu': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'repeat': options.repeat, 'trials': options.trials,
              'inputs': {
                  'filter_agg': 'arange(size, float32) - 17',
                  'sigma_clipped_mean': 'normal(seed=42), float32, '
                  'every 100th value *= 100'},
              'single_block_limit': Executable.single_block_limit,
              'elements_per_block': Executable.elements_per_block,
              'source_hashes': {path: hashlib.sha256(
                  (root / path).read_bytes()).hexdigest() for path in (
                      'cupy/_core/_fusion_bytecode.py',
                      'cupy/_core/_fusion_bytecode_kernel.py')}, 'rows': []}
    if options.baseline_frontend:
        report['baseline_frontend_sha256'] = hashlib.sha256(
            options.baseline_frontend.read_bytes()).hexdigest()
    before_fuse = (baseline_fuse(options.baseline_frontend)
                   if options.baseline_frontend else None)
    with cp.cuda.Stream(non_blocking=True):
        for size in (4096, 65536, 1048576, 8388608):
            host = np.random.default_rng(42).normal(size=size)
            host = host.astype('float32')
            host[::100] *= 100
            for function in (filter_agg, sigma_clipped_mean):
                data = (cp.arange(size, dtype=cp.float32) - 17
                        if function is filter_agg else cp.asarray(host))
                scheduled = fuse(function)
                control = fuse(function)
                plan, _ = control._select(control._arguments((data,), {}))
                old = plan.execute
                plan.execute = Executable(old.source, old.array_arg,
                                          old.buffers, old.dtype,
                                          block_limit=1)
                expected = (cp.count_nonzero(data % 2 == 0)
                            if function is filter_agg else function(data))
                cp.testing.assert_allclose(scheduled(data), expected,
                                           rtol=3e-5, atol=2e-6)
                cp.testing.assert_allclose(control(data), expected,
                                           rtol=3e-5, atol=2e-6)
                timings = measure({'single_block': control,
                                   'scheduled': scheduled}, (data,),
                                  options.repeat, options.trials)
                row = {'function': function.__name__, 'size': size,
                       'schedule': dict(scheduled.diagnostics['schedule']),
                       'timings': timings}
                report['rows'].append(row)
                print(function.__name__, size, row['schedule'], timings,
                      flush=True)
        for shape in ((64, 64), (4096, 64), (32768, 512)):
            data = cp.linspace(-1, 1, np.prod(shape), dtype=cp.float32)
            data = data.reshape(shape)
            scheduled = fuse(score)
            methods = {'current': scheduled}
            if before_fuse:
                methods['before'] = before_fuse(score)
            expected = score(data)
            for function in methods.values():
                cp.testing.assert_allclose(function(data), expected,
                                           rtol=3e-5, atol=2e-6)
            timings = measure(methods, (data,), options.repeat, options.trials)
            row = {'function': 'score', 'shape': shape, 'timings': timings,
                   'regions': len(scheduled.diagnostics['regions'])}
            report['rows'].append(row)
            print('score', shape, timings, flush=True)
    Path(options.json).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
