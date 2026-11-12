"""Analysis-time composition of fused regions and known implementations."""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

import cupy as cp
from cupy._core._fusion_bytecode import Unsupported, fuse
from examples.fusion.bytecode_iteration import filter_agg, sigma_clipped_mean


@pytest.fixture
def gpu():
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('CUDA device required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA device required')


def test_analysis_dispatch_never_launches_or_allocates(gpu):
    def function(a, b):
        return cp.sin(a) @ b + cp.cos(b)

    data = cp.ones((4, 4), dtype=cp.float32)
    with mock.patch.object(cp, 'empty',
                           side_effect=AssertionError('allocation')):
        with mock.patch.object(cp, 'RawKernel',
                               side_effect=AssertionError('compilation')):
            report = fuse(function).explain(data, data)
    assert [b['operation'] for b in report['boundaries']] == ['matmul']
    assert len(report['regions']) == 2


def test_linear_library_linear(gpu):
    def function(a, b):
        return cp.sin(a) @ b + cp.cos(b)

    a = cp.arange(16, dtype=cp.float32).reshape(4, 4) / 10
    b = a + 1
    compiled = fuse(function)
    for shift in (0., 0.5):
        cp.testing.assert_allclose(compiled(a + shift, b),
                                   function(a + shift, b),
                                   rtol=3e-5, atol=2e-6)
    assert compiled.diagnostics['cache_hit']


def test_composed_analysis_without_cuda():
    import json

    class ArrayMetadata:
        dtype = np.dtype('float32')
        ndim = 1

    def function(data):
        return filter_agg(cp.square(data)) + 1

    with mock.patch.object(cp, 'ndarray', ArrayMetadata):
        with mock.patch.object(cp, 'empty', side_effect=AssertionError):
            report = fuse(function).explain(ArrayMetadata())
    assert report['boundaries'][0]['plan']['kernel_launches'] == 1
    json.dumps(report)


@pytest.mark.parametrize('operation', [cp.dot, cp.tensordot, cp.linalg.solve,
                                       cp.linalg.inv, cp.fft.fft])
def test_registered_library_calls(gpu, operation):
    if operation in (cp.linalg.inv, cp.fft.fft):
        def function(a):
            return operation(cp.sin(a)) + 1
        data = cp.eye(4, dtype=cp.float64) + 0.25
        args = (data,)
    else:
        def function(a, b):
            return operation(cp.sin(a), b) + 1
        data = cp.eye(4, dtype=cp.float64) + 0.25
        args = (data, data + 1)
    cp.testing.assert_allclose(fuse(function)(*args), function(*args),
                               rtol=3e-5, atol=2e-6)


def test_library_method_forms(gpu):
    def function(data):
        return cp.sin(data.reshape(2, 8).transpose(1, 0).T)

    data = cp.arange(16, dtype=cp.float32)
    cp.testing.assert_array_equal(fuse(function)(data), function(data))


def test_library_calls_in_bounded_host_range(gpu):
    def function(data, weight, steps):
        for _ in range(steps):
            data = cp.sin(data) @ weight + data
        return data

    data = cp.ones((4, 4), dtype=cp.float32) / 10
    weight = cp.eye(4, dtype=cp.float32) / 2
    compiled = fuse(function)
    for steps in (2, 0, 3, 2):
        cp.testing.assert_allclose(compiled(data, weight, steps),
                                   function(data, weight, steps),
                                   rtol=3e-5, atol=2e-6)
    assert compiled.diagnostics['cache_hit']
    assert compiled.diagnostics['expanded_iterations'] == 2


def test_linear_cooperative_linear(gpu):
    def function(data):
        return filter_agg(cp.square(data)) + 1

    data = cp.arange(262147, dtype=cp.float32)
    compiled = fuse(function)
    expected = cp.count_nonzero(cp.square(data) % 2 == 0) + 1
    cp.testing.assert_array_equal(compiled(data), expected)
    boundary = compiled.diagnostics['boundaries'][0]
    assert boundary['operation'] == 'filter_agg'
    assert boundary['plan']['backend'] == 'scheduled_cuda'
    assert len(compiled.diagnostics['regions']) == 2
    assert boundary['plan']['schedule']['blocks'] > 1


def test_matrix_linear_cooperative_callee(gpu):
    def function(data, weight):
        return sigma_clipped_mean(cp.sin(data @ weight))

    data = cp.linspace(-1, 1, 65536, dtype=cp.float32).reshape(-1, 2)
    weight = cp.asarray([0.5, 0.25], dtype=cp.float32)
    compiled = fuse(function)
    cp.testing.assert_allclose(compiled(data, weight), function(data, weight),
                               rtol=3e-5, atol=2e-6)
    assert [b['operation'] for b in compiled.diagnostics['boundaries']] == [
        'matmul', 'sigma_clipped_mean']


@pytest.mark.parametrize('stride', [1, -2])
def test_device_loop_with_linear_math_and_local_rebinding(gpu, stride):
    def function(data, steps):
        for _ in range(steps):
            data = cp.sin(data) + 0.1
            if cp.mean(data) > 0.5:
                break
        return cp.mean(data)

    data = cp.linspace(-1, 1, 262147, dtype=cp.float32)[::stride]
    original = data.copy()
    compiled = fuse(function)
    for steps in (0, 3, 9):
        cp.testing.assert_allclose(compiled(data, steps),
                                   function(data, steps),
                                   rtol=3e-5, atol=2e-6)
    cp.testing.assert_array_equal(data, original)
    assert compiled.diagnostics['backend'] == 'scheduled_cuda'
    assert compiled.diagnostics['kernel_launches'] == 1


def test_source_free_matrix_and_compiled_callee(gpu):
    namespace = {'cp': cp, 'filter_agg': filter_agg}
    exec(compile('def function(a, b):\n'
                 '    return filter_agg(cp.square(a @ b)) + 1\n',
                 '<generated>', 'exec'), namespace)
    data = cp.ones((32768, 2), dtype=cp.float32)
    weight = cp.ones(2, dtype=cp.float32)
    assert int(fuse(namespace['function'])(data, weight)) == 32769


def test_mutation_order_and_overlapping_views(gpu):
    def function(data):
        previous = data + 1
        view = data[1:]
        data[1:] = data[:-1]
        data[0] += 10
        return previous, data, view

    data = cp.arange(20, dtype=cp.float32)
    reference = data.copy()
    expected = function(reference)
    actual = fuse(function)(data)
    for a, b in zip(actual, expected):
        cp.testing.assert_array_equal(a, b)
    assert actual[1] is data
    data[1] = 123
    assert int(actual[2][0]) == 123


def test_dynamic_slice_bounds_and_inplace(gpu):
    def function(data, start):
        previous = data * 2
        data += 1
        data[start:] = data[:-start]
        return previous, data

    compiled = fuse(function)
    for start in (1, 2):
        data = cp.arange(20, dtype=cp.float32)
        expected = function(data.copy(), start)
        for actual, reference in zip(compiled(data, start), expected):
            cp.testing.assert_array_equal(actual, reference)
    assert compiled.diagnostics['cache_hit']


def test_native_error_preserves_prior_write_without_replay(gpu):
    def function(data):
        data[0] += 1
        data[10000] = 0
        return data

    data = cp.zeros(4, dtype=cp.int64)
    with pytest.raises(IndexError):
        fuse(function)(data)
    assert int(data[0]) == 1


def test_noncontiguous_mutation_is_rejected(gpu):
    def function(data):
        data[0] = 7
        return data

    data = cp.arange(12)[::2]
    with pytest.raises(Unsupported, match='C-contiguous'):
        fuse(function)(data)
    assert int(data[0]) == 0


def test_rejection_precedes_mutation(gpu):
    calls = []

    def callback(data):
        calls.append(data)
        return data

    def function(data):
        data[0] = 7
        return callback(data)

    data = cp.zeros(4)
    with pytest.raises(Unsupported):
        fuse(function)(data)
    assert not calls
    assert int(data[0]) == 0


def test_excluded_recursion():
    def function(data):
        return function(data)

    with pytest.raises(Unsupported):
        fuse(function).explain(1)
