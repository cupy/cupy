"""Check saved measurements and render RST tables without importing CuPy."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import textwrap


def table(title, headers, rows):
    stream = io.StringIO()
    csv.writer(stream).writerows([headers, *rows])
    body = '\n'.join('   ' + line for line in stream.getvalue().splitlines())
    return f'.. csv-table:: {title}\n   :header-rows: 1\n\n{body}\n\n'


def number(value, digits=2):
    return '-' if value is None else f'{value:.{digits}f}'


def best_reuse(group):
    return min(row['gpu_elapsed_us'] for name, row in group.items()
               if name.startswith('eager_out') and row['status'] == 'ok')


def check(report):
    groups, keys = {}, set()
    trials = report['configuration']['trials']
    for row in report['rows']:
        key = row['profile'], row['case'], row['method']
        assert key not in keys, f'Duplicate measurement: {key}'
        keys.add(key)
        group = groups.setdefault(key[:2], {})
        group[row['method']] = row
        if row['status'] == 'unsupported':
            assert row['method'] == 'existing_fuse'
            assert row['error']
            continue
        assert row['status'] == 'ok'
        assert len(row['first_calls']) == row['variants']
        assert len(row['trials']) == trials
        assert row['repeat'] % row['variants'] == 0
        for metric in ('cpu_us', 'gpu_elapsed_us', 'input_gelem_s'):
            assert math.isfinite(row[metric]) and row[metric] > 0
        if row['method'] == 'prototype':
            assert row['prototype']['cache_hit']
    for group in groups.values():
        required = {'eager', 'eager_out', 'prototype', 'existing_fuse'}
        assert required <= group.keys()
    root = Path(__file__).resolve().parents[2]
    for path, expected in report['environment']['source_sha256'].items():
        actual = hashlib.sha256((root / path).read_bytes()).hexdigest()
        assert actual == expected, f'Source changed since measurement: {path}'
    return groups


def render(report, groups):
    env = report['environment']
    cfg = report['configuration']
    counts = Counter(row['status'] for row in report['rows'])
    profiles = list(dict.fromkeys(row['profile'] for row in report['rows']))
    cases = list(dict.fromkeys(row['case'] for row in report['rows']))
    title = 'Bytecode fusion benchmark results'
    text = f'{title}\n{"=" * len(title)}\n\n'
    text += (f'{len(cases)} named workloads, {len(profiles)} input profiles, '
             f'{len(groups)} case/profile combinations. '
             f'{counts["ok"]} measured configurations passed validation; '
             f'{counts["unsupported"]} existing-fuse refusals '
             'are recorded.\n\n')
    text += (f'GPU: {env["gpu"]}. Python {env["python"]}, '
             f'CuPy {env["cupy"]}, NumPy {env["numpy"]}, '
             f'CUDA runtime {env["runtime"]}. '
             f'Run started {env["started_utc"]}.\n\n')
    text += (f'Each number is the median of {cfg["trials"]} warmed batches. '
             'Batch lengths are adaptive and balanced across '
             'argument variants. '
             'GPU elapsed time includes host submission gaps. '
             'Gelem/s counts logical input elements, '
             'not device memory traffic. '
             'Full trials, first-call times, dispersion, cache metadata, '
             'source hashes and buffer sizes are in results.json; '
             'results.csv contains '
             'the flat measurements.\n\n')

    rows = []
    for profile in profiles:
        group = groups.get((profile, 'score'))
        if not group or group['prototype']['input_elements'] > 262144:
            continue
        e, r = group['eager'], group['eager_out']
        old, new = group['existing_fuse'], group['prototype']
        rows.append([profile, number(e['cpu_us']), number(r['cpu_us']),
                     number(old['cpu_us']), number(new['cpu_us']),
                     number(new['cpu_us'] / old['cpu_us'])])
    text += table('Score CPU submission time (microseconds)',
                  ['Profile', 'Eager', 'Reuse', 'Existing fuse', 'Prototype',
                   'Prototype / existing'], rows)

    rows = []
    for profile in profiles:
        group = groups.get((profile, 'score'))
        if not group:
            continue
        e, r, p = (group[name] for name in ('eager', 'eager_out', 'prototype'))
        rows.append([profile, f'{p["input_elements"]:,}',
                     number(e['gpu_elapsed_us']), number(r['gpu_elapsed_us']),
                     number(p['gpu_elapsed_us']),
                     f'{p["input_gelem_s"]:.3g}',
                     number(r['gpu_elapsed_us'] / p['gpu_elapsed_us'])])
    text += table('Score complete-call elapsed time (microseconds)',
                  ['Profile', 'Input elements', 'Eager', 'Reuse', 'Prototype',
                   'Prototype Gelem/s', 'Speedup / reuse'], rows)

    rows = []
    largest = {}
    for case in cases:
        choices = [group for (profile, name), group in groups.items()
                   if name == case
                   and profile not in ('transposed', 'strided')]
        if not choices:
            continue
        group = max(choices, key=lambda g: g['prototype']['input_elements'])
        largest[case] = group
        e, r, p = (group[name] for name in ('eager', 'eager_out', 'prototype'))
        constants = group.get('eager_out_constants')
        rows.append([
            case, number(e['gpu_elapsed_us'] / 1000, 3),
            number(r['gpu_elapsed_us'] / 1000, 3),
            number(constants['gpu_elapsed_us'] / 1000
                   if constants else None, 3),
            number(p['gpu_elapsed_us'] / 1000, 3),
            number(best_reuse(group) / p['gpu_elapsed_us'])])
    text += table('Largest contiguous inputs: complete-call milliseconds',
                  ['Case', 'Eager', 'Reuse + initialization',
                   'Reuse + constant initialization hoisted', 'Prototype',
                   'Speedup / best reuse'], rows)
    text += ('The constant baseline initializes persistent constant arrays '
             'outside timing. Ordinary reuse still initializes them per call. '
             'Both reuse baselines overwrite returned arrays on later calls. '
             'The constant baseline can retain additional full-size buffers. '
             'These are numerical comparisons, not equivalent allocation '
             'lifetimes or observable allocation effects.\n\n')

    rows = []
    for case in ('nested_numpy_scalars', 'nested_python_scalars',
                 'boolean_alternating', 'allocation_alternating',
                 'shape_alternating', 'instance_alternating'):
        if case not in largest:
            continue
        p = largest[case]['prototype']
        info = p['prototype']
        rows.append([case, p['variants'], info['plan_variants'],
                     info['analysed_functions']])
    text += table('Cache reuse on changing inputs',
                  ['Case', 'Argument variants', 'Cached plans',
                   'Analysed definitions'], rows)

    rows = []
    for profile in ('transposed', 'strided'):
        ratios = [(best_reuse(group) / group['prototype']['gpu_elapsed_us'],
                   case)
                  for (name, case), group in groups.items() if name == profile]
        if ratios:
            low, high = min(ratios), max(ratios)
            rows.append([profile, number(low[0]), low[1],
                         number(high[0]), high[1]])
    text += table('Speedup over best reuse on noncontiguous inputs',
                  ['Profile', 'Minimum', 'Case', 'Maximum', 'Case'], rows)

    text += ('Existing fuse includes its automatic new_fusion fallback. Its '
             'multi-stage schedules can be particularly slow on these '
             'reductions; large speedups over that path '
             'are workload-specific. '
             'Complete existing-fuse timings are preserved in JSON/CSV. '
             'Small-input CPU overhead and large-input throughput should be '
             'assessed separately.\n\n')
    if env['import_warnings']:
        text += ('The existing native build emitted these import warnings; '
                 'the artifacts were not rebuilt for this run::\n\n')
        text += ''.join(f'    {warning}\n'
                        for warning in env['import_warnings'])
    return '\n\n'.join(
        textwrap.fill(part, width=79)
        if '\n' not in part and not part.startswith(('..', '    '))
        else part for part in text.split('\n\n'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json', type=Path)
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    report = json.loads(options.json.read_text())
    groups = check(report)
    output = options.output or options.json.with_suffix('.rst')
    output.write_text(render(report, groups))
    # csv.writer uses CRLF by default. Normalize the saved artifact for Git,
    # without changing the timing runner or any measured values.
    csv_path = options.json.with_suffix('.csv')
    if csv_path.exists():
        csv_path.write_text(csv_path.read_text(), newline='\n')
    print(f'Checked {len(report["rows"])} rows; wrote {output}')


if __name__ == '__main__':
    main()
