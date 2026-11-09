"""Device loops and conditions in a single scalar-output kernel."""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

import cupy as cp
from cupy._core._fusion_bytecode import Unsupported, fuse


def filter_agg(data):
    total = 0
    for x in data:
        if x % 2 == 0:
            total += 1
    return total


def sigma_clipped_mean(data, sigma=3.0, max_iter=5):
    mask = cp.zeros(data.shape, dtype=bool)
    for _ in range(max_iter):
        valid_data = data[~mask]
        if len(valid_data) == 0:
            break
        mean = cp.mean(valid_data)
        std = cp.std(valid_data)
        new_mask = cp.abs(data - mean) > (sigma * std)
        if cp.all(mask == (mask | new_mask)):
            break
        mask = mask | new_mask
    return cp.mean(data[~mask])


class ArrayMetadata:
    """Permit explain tests on machines without a CUDA device."""
    dtype = np.dtype('float32')
    ndim = 1
    shape = (1000,)
    size = 1000


@pytest.mark.parametrize('function', [filter_agg, sigma_clipped_mean])
def test_explain_without_source_or_cuda(function):
    # Rebuild from a code object; no source file is available to the frontend.
    import types
    function = types.FunctionType(function.__code__, function.__globals__,
                                  function.__name__, function.__defaults__)
    with mock.patch.object(cp, 'ndarray', ArrayMetadata):
        compiled = fuse(function)
        report = compiled.explain(ArrayMetadata())
        assert report['backend'] == 'scheduled_cuda'
        assert report['schedule']['kind'] == 'not_selected'
        assert report['kernel_launches'] == 1
        assert '__global__ void iterative' in report['source']
        assert 'goto L' in report['source']
        assert compiled.explain(ArrayMetadata())['cache_hit']
        assert report['workspace_arrays'] == (
            0 if function.__name__ == 'filter_agg' else 3)


@pytest.fixture
def gpu():
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('CUDA device required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA device required')


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float32', 'float64'])
@pytest.mark.parametrize('size', [0, 1, 31, 256, 259, 10003])
def test_filter_agg(gpu, dtype, size):
    host = np.arange(size, dtype=dtype) - 17
    data = cp.asarray(host)
    compiled = fuse(filter_agg)
    actual = compiled(data)
    assert actual.shape == ()
    assert actual.dtype == np.dtype('int64')
    assert int(actual) == np.count_nonzero(host % 2 == 0)


@pytest.mark.parametrize('dtype', ['int32', 'float32', 'float64'])
@pytest.mark.parametrize('sigma,max_iter', [(3., 5), (1., 5), (2., 1),
                                            (3., 0), (3., -2), (0., 5)])
def test_sigma_clipped_mean(gpu, dtype, sigma, max_iter):
    host = np.r_[np.linspace(-1, 1, 1025), [5., -10., 50.]].astype(dtype)
    data = cp.asarray(host)
    expected = sigma_clipped_mean(data, sigma, max_iter)
    compiled = fuse(sigma_clipped_mean)
    actual = compiled(data, sigma, max_iter)
    assert actual.dtype == expected.dtype
    cp.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('host', [[], [1.], [2.] * 100,
                                  [float('nan'), 1.],
                                  [float('inf'), 1.], [0., 1., 100.]])
def test_sigma_edge_cases(gpu, host):
    data = cp.asarray(host, dtype=cp.float64)
    cp.testing.assert_allclose(fuse(sigma_clipped_mean)(data),
                               sigma_clipped_mean(data), equal_nan=True)


@pytest.mark.parametrize('function', [filter_agg, sigma_clipped_mean])
def test_strides_runtime_values_and_stream(gpu, function):
    compiled = fuse(function)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        for host in (np.arange(1001, dtype='float32'),
                     np.ones(1001, dtype='float32'),
                     np.arange(71, dtype='float32')):
            data = cp.asarray(host)[::-2]
            expected = function(data)
            actual = compiled(data)
            cp.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        assert compiled.diagnostics['cache_hit']
    stream.synchronize()


def test_changed_predicate_and_accumulator(gpu):
    def positive_sum(data):
        total = 0
        for x in data:
            if x > 0:
                total += 2
        return total

    data = cp.asarray([-3, 0, 1, 2, 7], dtype='int32')
    assert int(fuse(positive_sum)(data)) == 6


def test_runtime_parameters_and_cached_plan(gpu):
    data = cp.asarray([0., 1., 1., 2., 20.], dtype='float32')
    compiled = fuse(sigma_clipped_mean)
    for sigma, iterations in [(3., 0), (1., 5), (3., 5), (0., 9), (2., -1)]:
        cp.testing.assert_allclose(compiled(data, sigma, iterations),
                                   sigma_clipped_mean(data, sigma, iterations),
                                   rtol=2e-5, atol=2e-6)
    assert compiled.diagnostics['cache_hit']
    assert len(compiled._plans) == 1


@pytest.mark.parametrize('dtype', ['int64', 'float32', 'float64'])
def test_general_modulo_predicate(gpu, dtype):
    def function(data):
        total = 0
        for x in data:
            if x % -3 == 0:
                total += 1
        return total

    data = cp.arange(-20, 20, dtype=dtype)
    assert int(fuse(function)(data)) == int(cp.count_nonzero(data % -3 == 0))


@pytest.mark.parametrize('function', [filter_agg, sigma_clipped_mean])
@pytest.mark.parametrize('size', [513, 262147])
def test_one_kernel_in_cuda_graph(gpu, function, size):
    data = cp.arange(size, dtype='float32')
    compiled = fuse(function)
    compiled(data)  # Compile and populate the memory pool before capture.
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        output = compiled(data)
        graph = stream.end_capture()
        graph.launch()
    stream.synchronize()
    cp.testing.assert_allclose(output, function(data), rtol=2e-5, atol=2e-6)
    # Inspect the actual captured CUDA graph, rather than trusting diagnostics.
    dot = graph.debug_dot_str()
    assert dot.count('iterative') == 1, dot
    assert dot.count('shape="octagon"') == 1, dot


@pytest.mark.parametrize('function', [filter_agg, sigma_clipped_mean])
def test_schedule_changes_with_shape(gpu, function):
    compiled = fuse(function)
    for size in (513, 262147, 513, 1048583):
        data = cp.arange(size, dtype='float64')[::-2]
        actual = compiled(data)
        expected = (cp.count_nonzero(data % 2 == 0)
                    if function is filter_agg else function(data))
        cp.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        schedule = compiled.diagnostics['schedule']
        if data.size <= 16384:
            assert schedule['blocks'] == 1
        elif cp.cuda.Device().attributes.get('CooperativeLaunch', 0):
            assert schedule['blocks'] > 1
            assert schedule['cooperative']
        assert compiled.diagnostics['kernel_launches'] == 1
    assert len(compiled._plans) == 1


@pytest.mark.parametrize('sigma,iterations', [(1., 5), (3., 5), (0., 5),
                                              (3., 0), (3., -1)])
def test_cooperative_mask_updates(gpu, sigma, iterations):
    host = np.random.default_rng(42).normal(size=262147).astype('float32')
    host[::100] *= 100
    compiled = fuse(sigma_clipped_mean)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        data = cp.asarray(host)
        actual = compiled(data, sigma, iterations)
        expected = sigma_clipped_mean(data, sigma, iterations)
        cp.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    stream.synchronize()


def test_score_bypasses_iterative_scheduler(gpu):
    from examples.fusion.bytecode_prototype import score
    from cupy._core._fusion_bytecode_kernel import Executable

    data = cp.ones((64, 64), dtype=cp.float32)
    compiled = fuse(score)
    with mock.patch.object(Executable, '_launch',
                           side_effect=AssertionError('scheduler called')):
        compiled(data)
        actual = compiled(data)
    cp.testing.assert_allclose(actual, score(data), rtol=3e-5, atol=2e-6)
    assert compiled.diagnostics['cache_hit']
    assert 'schedule' not in compiled.diagnostics


def test_no_cooperative_launch_support(gpu):
    from cupy._core._fusion_bytecode_kernel import Executable

    data = cp.arange(100003, dtype='int32')
    compiled = fuse(filter_agg)
    plan, _ = compiled._select((data,))
    # The capacity cache records the one-block fallback on unsupported GPUs.
    plan.execute.devices[cp.cuda.runtime.getDevice()] = 1
    with mock.patch.object(Executable, '_kernel',
                           wraps=plan.execute._kernel) as select:
        actual = compiled(data)
    assert int(actual) == 50002
    assert compiled.diagnostics['schedule']['blocks'] == 1
    assert all(not call.args[1] for call in select.call_args_list)


def test_recurrent_element_accumulator_rejected():
    def function(data):
        total = 0
        for x in data:
            if total > 0:
                total += 1
        return total

    with mock.patch.object(cp, 'ndarray', ArrayMetadata):
        with pytest.raises(Unsupported, match='accumulators'):
            fuse(function).explain(ArrayMetadata())


def test_nonzero_seed_rejected():
    def function(data):
        total = 2
        for x in data:
            total += 1
        return total

    with mock.patch.object(cp, 'ndarray', ArrayMetadata):
        with pytest.raises(Unsupported, match='zero-initialized'):
            fuse(function).explain(ArrayMetadata())


def test_uninitialized_local_rejected():
    def function(data):
        for i in range(5):
            if cp.all(data > 0):
                value = cp.mean(data)
        return value

    with mock.patch.object(cp, 'ndarray', ArrayMetadata):
        with pytest.raises(Unsupported, match='Uninitialized local'):
            fuse(function).explain(ArrayMetadata())


def test_element_break_rejected():
    def function(data):
        total = 0
        for x in data:
            if x > 0:
                break
            total += 1
        return total

    with mock.patch.object(cp, 'ndarray', ArrayMetadata):
        with pytest.raises(Unsupported, match='break in element'):
            fuse(function).explain(ArrayMetadata())


def test_element_accumulator_reset_rejected():
    def function(data):
        total = 0
        for x in data:
            total += 1
            if x > 0:
                total = 0
        return total

    with mock.patch.object(cp, 'ndarray', ArrayMetadata):
        with pytest.raises(Unsupported, match='additive accumulator updates'):
            fuse(function).explain(ArrayMetadata())
