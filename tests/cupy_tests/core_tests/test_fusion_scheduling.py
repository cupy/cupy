"""Literal powers and device-dependent scheduling of tracing reductions."""
from __future__ import annotations

from unittest import mock

import pytest

import cupy as cp
from cupy._core import _fusion_kernel, new_fusion
from cupy._core._fusion_bytecode import fuse


@pytest.fixture
def gpu():
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('CUDA device required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA device required')


@pytest.mark.parametrize('dtype', [
    'bool', 'uint8', 'uint64', 'int8', 'int64',
    'float32', 'float64', 'complex64'])
def test_literal_square_preserves_dtype(gpu, dtype):
    def function(x):
        return x**2

    x = cp.arange(257).astype(dtype)
    compiled = fuse(function)
    actual, expected = compiled(x), function(x)
    assert actual.dtype == expected.dtype
    cp.testing.assert_array_equal(actual, expected)
    operations = compiled.diagnostics['regions'][0]['operations']
    assert operations == ['square']


def test_computed_square_and_runtime_exponent(gpu):
    def literal(x):
        return (x - 1)**2

    def runtime(x, exponent):
        return x**exponent

    x = cp.linspace(-2, 2, 257)
    compiled = fuse(literal)
    cp.testing.assert_allclose(compiled(x), literal(x))
    assert compiled.diagnostics['regions'][0]['operations'] == [
        'subtract', 'square']
    compiled = fuse(runtime)
    for exponent in (2, 3):
        cp.testing.assert_allclose(compiled(x, exponent), runtime(x, exponent))
        assert compiled.diagnostics['regions'][0]['operations'] == ['power']


def test_squared_boolean_expression(gpu):
    def function(x):
        return (x > 0)**2

    x = cp.linspace(-1, 1, 257)
    actual, expected = fuse(function)(x), function(x)
    assert actual.dtype == expected.dtype
    cp.testing.assert_array_equal(actual, expected)


def test_literal_square_after_reduced_mean(gpu):
    def function(x, n):
        mean = cp.sum(x, axis=0) / n
        return cp.sum((x - mean)**2, axis=0)

    x = cp.linspace(.1, .8, 129 * 33).reshape(129, 33)
    compiled = fuse(function)
    cp.testing.assert_allclose(compiled(x, 129), function(x, 129))
    operations = [op for region in compiled.diagnostics['regions']
                  for op in region['operations']]
    assert 'square' in operations
    assert 'power' not in operations


def test_square_after_boolean_reduction(gpu):
    def function(x):
        return cp.all(x)**2

    x = cp.ones(257, dtype=bool)
    actual, expected = fuse(function)(x), function(x)
    assert actual.dtype == expected.dtype
    cp.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('dtype', ['bool', 'float32', 'float64'])
def test_square_after_library_boundary(gpu, dtype):
    def mean_squared(x):
        return cp.mean(x, axis=1)**2

    def transposed_squared(x):
        return cp.transpose(x)**2

    x = cp.arange(17 * 13).reshape(17, 13).astype(dtype)
    for function in (mean_squared, transposed_squared):
        compiled = fuse(function)
        actual, expected = compiled(x), function(x)
        assert actual.dtype == expected.dtype
        cp.testing.assert_allclose(actual, expected, rtol=3e-6)
        assert compiled.diagnostics['regions'][-1]['operations'] == ['square']


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_literal_square_in_device_loop(gpu, dtype):
    def function(data, iterations):
        for _ in range(iterations):
            data = data**2
            if cp.mean(data) > 1000:
                break
        return cp.mean(data)

    data = cp.linspace(.1, .8, 257, dtype=dtype)
    expected = function(data, 3)
    compiled = fuse(function)
    actual = compiled(data, 3)
    assert compiled.diagnostics['backend'] == 'scheduled_cuda'
    assert actual.dtype == expected.dtype
    cp.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-12)


@pytest.mark.parametrize('sms,cooperative,split', [
    (1, True, False), (4, True, True), (4, False, True)])
def test_schedule_uses_device_limits(gpu, sms, cooperative, split):
    def function(x):
        return cp.sum(cp.sin(x)) + 1

    x = cp.linspace(.1, .8, 262144)
    compiled = new_fusion.Fusion(function)
    original = _fusion_kernel._cuda_compile
    with mock.patch.object(_fusion_kernel, '_get_device_limits',
                           return_value=(sms, cooperative)):
        with mock.patch.object(_fusion_kernel, '_cuda_compile',
                               wraps=original) as compiler:
            actual = compiled(x)
    cp.testing.assert_allclose(actual, function(x), rtol=2e-12)
    names = [call.args[1] for call in compiler.call_args_list]
    assert any('_stage_' in name for name in names) == split


def test_grid_scales_with_device_and_caps_small_work(gpu):
    def function(x):
        return cp.sum(cp.exp(x), axis=1) + 1

    original = _fusion_kernel._cuda_compile
    launches = []

    class Kernel:
        def __init__(self, function):
            self.function = function
            self.ptr = function.ptr

        def linear_launch(self, size, args, shared_mem, block_size, **kwargs):
            launches.append((size, block_size))
            return self.function.linear_launch(
                size, args, shared_mem, block_size, **kwargs)

    for rows in (4096, 4):
        x = cp.linspace(.1, .8, rows * 32).reshape(rows, 32)
        sizes = []
        compiled = new_fusion.Fusion(function)
        for sms in (1, 4):
            with mock.patch.object(_fusion_kernel, '_get_device_limits',
                                   return_value=(sms, True)):
                with mock.patch.object(_fusion_kernel, '_cuda_compile',
                                       side_effect=lambda *a: Kernel(
                                           original(*a))):
                    cp.testing.assert_allclose(compiled(x), function(x))
            sizes.append(launches[-1][0])
        assert sizes[1] == (4 * sizes[0] if rows == 4096 else sizes[0])
        if rows == 4:
            assert sizes[0] == launches[-1][1]


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int64'])
@pytest.mark.parametrize('operation', [cp.sum, cp.prod, cp.min, cp.max,
                                       cp.all, cp.any])
def test_large_scalar_reductions(gpu, dtype, operation):
    def function(x):
        return operation(x + 0)

    x = cp.ones(262144, dtype=dtype)
    x[13] = 0
    compiled = new_fusion.Fusion(function)
    actual, expected = compiled(x), function(x)
    assert actual.dtype == expected.dtype
    cp.testing.assert_array_equal(actual, expected)


def test_staged_multi_axis_strides_and_tuple(gpu):
    def function(x):
        return (cp.sum(cp.sin(x), axis=(0, 2)),
                cp.sum(x, axis=(1,)))

    compiled = new_fusion.Fusion(function)
    base = cp.linspace(.1, .8, 129 * 129 * 33).reshape(129, 129, 33)
    for x in (base, base[::-1, ::2, ::-1], base[:65]):
        for actual, expected in zip(compiled(x), function(x)):
            cp.testing.assert_allclose(actual, expected, rtol=2e-12)


def test_staged_cache_handles_c_and_fortran_layouts(gpu):
    def function(x):
        return cp.sum(cp.sin(x), axis=0) + 1

    x = cp.sin(cp.arange(1024 * 257).reshape(1024, 257))
    f = cp.asfortranarray(x)
    # Both own their storage: the tracer sees identical dtype/rank/base keys,
    # so the staged CUDA signature cache must distinguish their layouts.
    assert x.base is None and f.base is None
    compiled = new_fusion.Fusion(function)
    for data in (x, f, x):
        cp.testing.assert_allclose(compiled(data), function(data), rtol=2e-12)


def test_staged_output_and_input_mutation(gpu):
    def function(x):
        total = cp.sum(x)
        x *= 2
        return total + cp.sum(x)

    x = cp.arange(262144, dtype=cp.float64)
    expected = function(x.copy())
    actual_input = x.copy()
    cp.testing.assert_allclose(new_fusion.Fusion(function)(actual_input),
                               expected)
    cp.testing.assert_array_equal(actual_input, x * 2)


def test_staged_reduction_cuda_graph(gpu):
    if cp.cuda.runtime.is_hip:
        pytest.skip('CUDA graph capture required')

    def function(x):
        return cp.sum(cp.sin(x)) + 1

    x = cp.linspace(.1, .8, 262144)
    compiled = new_fusion.Fusion(function)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        compiled(x)
    stream.synchronize()
    with stream:
        stream.begin_capture()
        actual = compiled(x)
        graph = stream.end_capture()
        x.fill(.5)
        graph.launch(stream)
    stream.synchronize()
    cp.testing.assert_allclose(actual, function(x), rtol=2e-12)
