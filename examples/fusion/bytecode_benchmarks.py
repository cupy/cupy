"""Benchmark bytecode fusion against allocating and explicit-out eager CuPy.

Run from a built checkout, for example:
    python -m examples.fusion.bytecode_benchmarks --profiles small large xlarge

JSON and CSV are checkpointed after each case. Numerical mismatches and
prototype failures are fatal; only known existing-fuse refusals are recorded
as unsupported. No timings include validation, input/scratch setup, or first
compilation. CUDA events measure whole calls including host submission gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types
import warnings

import numpy as np

with warnings.catch_warnings(record=True) as _import_warnings:
    warnings.simplefilter('always', RuntimeWarning)
    import cupy as cp
    from cupy._core._fusion_bytecode import fuse
    from cupy._core.fusion import Fusion
    from examples.fusion._bytecode_benchmark_cases import cases


PROFILES = {
    'micro': ((1, 1), 'C'),
    'tiny': ((8, 32), 'C'),
    'small': ((64, 64), 'C'),
    'medium': ((4096, 64), 'C'),
    'large': ((32768, 512), 'C'),
    'wide': ((128, 131072), 'C'),
    'transposed': ((32768, 512), 'transpose'),
    'strided': ((32768, 512), 'stride2'),
    'xlarge': ((131072, 1024), 'C'),
}
ROOT = Path(__file__).resolve().parents[2]


def array_results(result):
    if type(result) is cp.ndarray:
        return (result,)
    if type(result) is tuple:
        return tuple(a for item in result for a in array_results(item))
    raise AssertionError(f'Unexpected result type: {type(result)}')


def validate(actual, expected):
    outputs, references = array_results(actual), array_results(expected)
    assert len(outputs) == len(references)
    error = 0.0
    for output, reference in zip(outputs, references):
        assert output.shape == reference.shape
        assert output.dtype == reference.dtype
        matches = bool(cp.allclose(output, reference, rtol=3e-5, atol=2e-6))
        error = max(error, float(cp.max(cp.abs(output - reference))))
        if not matches:
            raise AssertionError('Numerical mismatch; '
                                 f'max absolute error {error}')
    return error


def scratch_bytes(function):
    """Count arrays owned by reference closures, not runtime inputs."""
    seen, arrays = set(), []

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if type(value) is cp.ndarray:
            arrays.append(value.nbytes)
        elif type(value) is types.FunctionType:
            for cell in value.__closure__ or ():
                visit(cell.cell_contents)
        elif type(value) is dict:
            for child in value.values():
                visit(child)

    visit(function)
    return sum(arrays)


def prototype_info(function):
    if not hasattr(function, '_plans'):
        return None
    report = function.diagnostics
    return {
        'plan_variants': sum(map(len, function._plans.values())),
        'analysed_functions': len(function._analyzer.functions),
        'cache_hit': report['cache_hit'],
        'region_count': len(report['regions']),
        'allocation_count': len(report['allocations']),
        'backends': [r['backend'] for r in report['regions']],
        'operations': [r['operations'] for r in report['regions']],
        'callees': report['callees'],
        'guards': report['guards'],
        'analysis_seconds': report['analysis_seconds'],
    }


def batch(function, variants, count):
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    before = time.perf_counter_ns()
    for index in range(count):
        function(*variants[index % len(variants)])
    cpu_us = (time.perf_counter_ns() - before) / count / 1000
    stop.record()
    stop.synchronize()
    return {'cpu_us': cpu_us,
            'gpu_elapsed_us': cp.cuda.get_elapsed_time(start, stop)
            * 1000 / count}


def legacy_refusal(exc):
    return (isinstance(exc, (TypeError, NotImplementedError, AttributeError))
            or type(exc) is Exception and str(exc) in (
                'Shape mismatch', "Can't cast to bool"))


@dataclass
class Method:
    name: str
    function: object
    status: str = 'ok'
    error: str | None = None
    first_calls: list = field(default_factory=list)
    max_abs_error: float = 0
    samples: list = field(default_factory=list)
    repeat: int = 0
    scratch: int = 0


def benchmark(case, options):
    reused = case.reused()
    methods = [Method('eager', case.function),
               Method('eager_out', reused, scratch=scratch_bytes(reused)),
               Method('existing_fuse', Fusion(case.function)),
               Method('prototype', fuse(case.function))]
    if case.constant_factory is not None:
        constants = case.reused(constants=True)
        methods.insert(2, Method('eager_out_constants', constants,
                                 scratch=scratch_bytes(constants)))
    for variant, args in enumerate(case.arguments):
        expected = case.function(*args)
        cp.cuda.get_current_stream().synchronize()
        for method in methods:
            if method.status != 'ok':
                continue
            try:
                before = time.perf_counter()
                actual = method.function(*args)
                cp.cuda.get_current_stream().synchronize()
                first_ms = (time.perf_counter() - before) * 1000
            except Exception as exc:
                if method.name != 'existing_fuse' or not legacy_refusal(exc):
                    raise
                method.status = 'unsupported'
                method.error = f'{type(exc).__name__}: {exc}'
                continue
            # Validation errors never become "unsupported" rows.
            method.max_abs_error = max(method.max_abs_error,
                                       validate(actual, expected))
            info = prototype_info(method.function)
            method.first_calls.append({
                'variant': variant, 'wall_ms': first_ms,
                'plan_cache_hit': info['cache_hit'] if info else None})
            del actual
        del expected

    active = [m for m in methods if m.status == 'ok']
    for method in active:
        pilot = batch(method.function, case.arguments, len(case.arguments))
        per_call_us = max(pilot['cpu_us'], pilot['gpu_elapsed_us'], 0.1)
        repeat = max(options.min_repeat, min(
            options.max_repeat, int(options.target_ms * 1000 / per_call_us)))
        # Give every variant equal representation in each batch.
        method.repeat = math.ceil(repeat / len(case.arguments)) * len(
            case.arguments)
        warmup = max(len(case.arguments), min(
            options.warmup, int(options.target_ms * 1000 / per_call_us)))
        batch(method.function, case.arguments, warmup)

    prototype = next(m.function for m in active if m.name == 'prototype')
    prepared = prototype_info(prototype)
    for trial in range(options.trials):
        # Rotate order so a method is not always first or last in a trial.
        offset = trial % len(active)
        order = active[offset:] + active[:offset]
        for method in order:
            sample = batch(method.function, case.arguments, method.repeat)
            method.samples.append(dict(sample, trial=trial,
                                       order=order.index(method)))
        current = prototype_info(prototype)
        assert current['plan_variants'] == prepared['plan_variants']
        assert current['analysed_functions'] == prepared['analysed_functions']
        assert current['cache_hit'], 'New analysis during warm timing'

    x_values = [args[case.x_index] for args in case.arguments]
    mean_elements = statistics.mean(x.size for x in x_values)
    rows = []
    for method in methods:
        row = {'case': case.name, 'feature': case.feature,
               'profile': options.profile, 'method': method.name,
               'dtype': str(x_values[0].dtype),
               'input_shapes': [list(x.shape) for x in x_values],
               'input_strides': [list(x.strides) for x in x_values],
               'input_elements': mean_elements,
               'variants': len(case.arguments), 'status': method.status,
               'error': method.error, 'max_abs_error': method.max_abs_error,
               'first_calls': method.first_calls,
               'scratch_bytes': method.scratch,
               'repeat': method.repeat, 'trials': method.samples}
        if method.status == 'ok':
            for metric in ('cpu_us', 'gpu_elapsed_us'):
                values = [sample[metric] for sample in method.samples]
                row[metric] = statistics.median(values)
                row[metric + '_min'] = min(values)
                row[metric + '_max'] = max(values)
                row[metric + '_stdev'] = statistics.pstdev(values)
            row['input_gelem_s'] = mean_elements / (
                row['gpu_elapsed_us'] * 1000)
            info = prototype_info(method.function)
            if info:
                row['prototype'] = info
            if method.name == 'existing_fuse':
                row['backend'] = ('new_fusion'
                                  if method.function.new_fusion is not None
                                  else 'legacy_fusion')
        rows.append(row)
    by_name = {row['method']: row for row in rows}
    for row in rows:
        if row['status'] == 'ok':
            for baseline in ('eager', 'eager_out', 'eager_out_constants',
                             'existing_fuse'):
                other = by_name.get(baseline)
                if other is not None and other['status'] == 'ok':
                    row[f'speedup_vs_{baseline}'] = (
                        other['gpu_elapsed_us'] / row['gpu_elapsed_us'])
    return rows


def make_input(profile, dtype):
    shape, layout = PROFILES[profile]
    count = math.prod(shape)
    if layout == 'transpose':
        return cp.linspace(-0.75, 0.85, count, dtype=dtype).reshape(
            shape[::-1]).T
    if layout == 'stride2':
        return cp.linspace(-0.75, 0.85, count * 2, dtype=dtype).reshape(
            shape[0], shape[1] * 2)[:, ::2]
    return cp.linspace(-0.75, 0.85, count, dtype=dtype).reshape(shape)


def source_hashes():
    paths = ('cupy/_core/_fusion_bytecode.py',
             'examples/fusion/bytecode_benchmarks.py',
             'examples/fusion/_bytecode_benchmark_cases.py')
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in paths}


def environment(options):
    properties = cp.cuda.runtime.getDeviceProperties(options.device)
    return {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': source_hashes(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'cupy': cp.__version__, 'numpy': np.__version__,
        'native_fusion_module': sys.modules[Fusion.__module__].__file__,
        'device': options.device, 'gpu': properties['name'].decode(),
        'gpu_memory_bytes': properties['totalGlobalMem'],
        'compute_capability': [properties['major'], properties['minor']],
        'runtime': cp.cuda.runtime.runtimeGetVersion(),
        'driver': cp.cuda.runtime.driverGetVersion(),
        'import_warnings': list(dict.fromkeys(
            str(w.message) for w in _import_warnings)),
    }


def write_results(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)
    fields = ('case', 'feature', 'profile', 'dtype', 'input_shapes', 'method',
              'status', 'cpu_us', 'gpu_elapsed_us', 'input_gelem_s',
              'speedup_vs_eager', 'speedup_vs_eager_out',
              'speedup_vs_eager_out_constants',
              'speedup_vs_existing_fuse', 'repeat', 'scratch_bytes',
              'max_abs_error', 'error')
    with path.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['rows'])


def parse_options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', nargs='+', choices=PROFILES,
                        default=['small', 'large', 'xlarge'])
    parser.add_argument('--cases', nargs='+', help='Names from --list-cases')
    parser.add_argument('--list-cases', action='store_true')
    parser.add_argument('--dtype', choices=['float32', 'float64'],
                        default='float32')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--json', type=Path,
                        default=ROOT / 'benchmarks/fusion_bytecode'
                        / 'results.json')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--min-repeat', type=int, default=3)
    parser.add_argument('--max-repeat', type=int, default=300)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--target-ms', type=float, default=30)
    options = parser.parse_args()
    if (min(options.trials, options.min_repeat, options.max_repeat,
            options.warmup, options.target_ms) <= 0
            or options.min_repeat > options.max_repeat):
        parser.error('Timing settings must be positive and min-repeat <= max')
    return options


def main():
    options = parse_options()
    for message in dict.fromkeys(str(w.message) for w in _import_warnings):
        print(f'CuPy import warning: {message}', file=sys.stderr)
    with cp.cuda.Device(options.device), cp.cuda.Stream(non_blocking=True):
        if options.list_cases:
            for case in cases(cp.empty((8, 64), dtype=options.dtype)):
                print(f'{case.name:28} {case.feature}')
            return
        metadata = environment(options)
        configuration = {key: getattr(options, key) for key in (
            'dtype', 'device', 'trials', 'min_repeat', 'max_repeat',
            'warmup', 'target_ms')}
        report = {'environment': metadata, 'configuration': configuration,
                  'methodology': {
                      'event_timing': 'whole calls with host submission gaps',
                      'throughput': 'logical input elements / event elapsed',
                      'reused_output_lifetime': 'overwritten by later calls',
                      'constructor_initialization': 'performed every call',
                      'constant_baseline': 'eager_out_constants initializes '
                      'non-escaping constant arrays outside timing',
                      'scratch_setup': 'outside timing',
                      'first_calls': 'not guaranteed cold compilation caches'},
                  'rows': []}
        if options.resume and options.json.exists():
            report = json.loads(options.json.read_text())
            old = report['environment']
            assert old['source_sha256'] == metadata['source_sha256']
            assert old['gpu'] == metadata['gpu']
            assert report['configuration'] == configuration
            report.setdefault('resumed_utc', []).append(
                metadata['started_utc'])
        completed = {(row['profile'], row['case']) for row in report['rows']}
        selected = set(options.cases or ())
        seen = set()
        for profile in options.profiles:
            options.profile = profile
            x = make_input(profile, options.dtype)
            for case in cases(x):
                seen.add(case.name)
                if selected and case.name not in selected:
                    continue
                if (profile, case.name) in completed:
                    continue
                print(f'Running {profile}/{case.name}', flush=True)
                rows = benchmark(case, options)
                report['rows'].extend(rows)
                write_results(report, options.json)
                timing = ', '.join(
                    f'{r["method"]}: {r.get("gpu_elapsed_us", 0):.2f} us'
                    if r['status'] == 'ok' else f'{r["method"]}: unsupported'
                    for r in rows)
                print(f'  {timing}', flush=True)
            del x
        unknown = selected - seen
        if unknown:
            raise ValueError(f'Unknown/unavailable cases: {sorted(unknown)}')
        print(f'Saved {len(report["rows"])} rows to {options.json}',
              flush=True)


if __name__ == '__main__':
    main()
