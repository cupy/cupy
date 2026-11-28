"""Contract tests for the private bytecode fusion prototype."""
from __future__ import annotations

import json
from unittest import mock

import numpy as np
import pytest

import cupy as cp
from cupy._core._fusion_bytecode import Unsupported, fuse


def score(x):
    energy = (cp.exp(-abs(x)) * cp.sin(x)**2).sum(axis=1)
    mass = (1 / (1 + cp.exp(-x))).sum(axis=1)
    return cp.sqrt(energy) / (mass + 1)


def test_explain_never_executes_user_body_or_cuda():
    calls = []

    def effect(x):
        calls.append(x)
        return x

    compiled = fuse(effect)
    with pytest.raises(Unsupported):
        compiled.explain(1)
    assert calls == []
    assert compiled.diagnostics['status'] == 'rejected'


def test_source_less_function_and_scalar_conversion():
    namespace = {}
    exec(compile('def f(x):\n return x + 1\n', '<generated>', 'exec'),
         namespace)
    compiled = fuse(namespace['f'])
    report = compiled.explain(3)
    assert report['scalar_contract']['int'] == 'int64'
    assert report['regions'][0]['operations'] == ['add']
    assert not report['cache_hit']
    assert compiled.explain(7)['cache_hit']
    with pytest.raises(OverflowError):
        compiled.explain(2**100)
    json.dumps(report)


@pytest.mark.parametrize('name', [
    'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh',
    'arcsinh', 'arctanh', 'exp2', 'expm1', 'log1p',
    'isnan', 'isinf', 'isfinite', 'signbit'])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_namespace_ufuncs(gpu, name, dtype):
    operation = getattr(cp, name)

    def function(x):
        return operation(x)

    data = cp.linspace(-.8, .8, 257, dtype=dtype)
    if name in ('isnan', 'isinf', 'isfinite', 'signbit'):
        data[:4] = cp.asarray([cp.nan, cp.inf, -cp.inf, -0.], dtype=dtype)
    actual, expected = fuse(function)(data), function(data)
    assert actual.dtype == expected.dtype
    cp.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-12)


@pytest.mark.parametrize('name', ['arctan2', 'atan2', 'hypot', 'copysign',
                                  'logical_and', 'logical_or', 'logical_xor'])
def test_binary_namespace_ufuncs(gpu, name):
    operation = getattr(cp, name)

    def function(x, y):
        return operation(x, y)

    x = cp.linspace(-2, 2, 257)
    cp.testing.assert_allclose(fuse(function)(x, 1.), function(x, 1.))


def test_added_ufuncs_in_existing_cupyx_helpers(gpu):
    from cupyx.scipy.signal import _bsplines
    from cupy._manipulation.add_remove import _unique_update_mask_equal_nan

    lam = cp.linspace(.1, 2., 257)
    compiled = fuse(_bsplines._coeff_smooth.func)
    for actual, expected in zip(compiled(lam), _bsplines._coeff_smooth(lam)):
        cp.testing.assert_allclose(actual, expected, rtol=2e-12)
    x = cp.asarray([0., cp.nan, 2., cp.nan])
    mask = cp.ones(4, dtype=bool)
    compiled = fuse(_unique_update_mask_equal_nan.func)
    assert compiled(mask, x) is None
    cp.testing.assert_array_equal(mask, cp.asarray([True, False, True, False]))


def test_added_ufunc_output_and_multioutput_rejected():
    def with_output(x):
        return cp.arctan(x, out=x)

    def multioutput(x):
        return cp.modf(x)

    with pytest.raises(Unsupported, match='Unsupported keyword'):
        fuse(with_output).explain(1.)
    with pytest.raises(Unsupported):
        fuse(multioutput).explain(1.)


def test_numpy_arrays_rejected():
    compiled = fuse(lambda x: x + 1)
    with pytest.raises(Unsupported, match='ndarray'):
        compiled.explain(np.arange(3))


def test_unknown_untaken_branch_not_parsed():
    calls = []

    def callback(x):
        calls.append(x)
        return x

    def function(x, flag):
        if flag:
            return x + 1
        return callback(x)

    compiled = fuse(function)
    assert compiled.explain(1, True)['regions'][0]['operations'] == ['add']
    with pytest.raises(Unsupported):
        compiled.explain(1, False)
    assert calls == []


def test_identity_analysis_cache_and_explicit_invalidation():
    def helper(x):
        return x + 1

    def root(x, flag):
        if flag:
            return helper(x)
        return helper(x) * 2

    compiled = fuse(root)
    compiled.explain(2, True)
    compiled.explain(2, False)
    assert len(compiled._analyzer.functions) == 2
    assert compiled._analyzer.hits >= 2
    assert compiled._analyzer.functions[id(helper)].function is helper

    def replacement(x):
        return x - 1

    helper.__code__ = replacement.__code__
    # Identity-only analysis intentionally preserves the old definition.
    assert compiled.explain(2, True)['regions'][0]['operations'] == ['add']
    compiled.clear_cache()
    report = compiled.explain(2, True)
    assert report['regions'][0]['operations'] == ['subtract']


def test_descriptors_are_rejected_without_execution():
    class Model:
        @property
        def scale(self):
            raise AssertionError('property executed')

        @fuse
        def apply(self, x):
            return x * self.scale

    model = Model()
    with pytest.raises(Unsupported, match='Descriptor'):
        Model.apply.explain(model, 2)


def test_unsupported_calls_rejected_before_allocations():
    def function(n):
        x = cp.zeros(n)
        print('must not execute')
        return x

    # Checking the generated plan must not call even registered allocation.
    with mock.patch('builtins.print', side_effect=AssertionError('executed')):
        with pytest.raises(Unsupported):
            fuse(function).explain(4)


def test_arithmetic_is_not_evaluated_during_explain():
    def function(x):
        return x / 0

    report = fuse(function).explain(np.float32(1))
    assert report['regions'][0]['operations'] == ['true_divide']


def test_boolean_not_and_unselected_mutation():
    def function(x, flag):
        if not flag:
            return x + 1
        x[0] = 42
        return x

    compiled = fuse(function)
    assert compiled.explain(2, False)['status'] == 'analysed'
    with pytest.raises(Unsupported):
        compiled.explain(2, True)


def test_positional_output_rejected_without_execution():
    def function(x, out):
        return cp.add(x, x, out)

    with pytest.raises(Unsupported, match='positional output'):
        fuse(function).explain(1, 2)


def test_keyword_only_and_default_binding():
    def function(x, *, scale=2):
        return x * scale

    compiled = fuse(function)
    assert compiled.explain(2)['regions'][0]['operations'] == ['multiply']
    assert compiled.explain(x=3, scale=4)['cache_hit']


def test_large_integer_constant_is_not_checked_as_an_input():
    def function(x):
        return x + 2**100

    assert fuse(function).explain(2)['status'] == 'analysed'


def test_combined_local_instructions():
    def function(x, y):
        a, b = x + y, x * y
        return a + b

    assert fuse(function).explain(2, 3)['regions'][0]['operations'] == [
        'add', 'multiply', 'add']


@pytest.fixture
def gpu():
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('CUDA device required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA device required')


@pytest.mark.parametrize('shape', [(8, 16), (64, 64), (1024, 128)])
def test_score(gpu, shape):
    x = cp.linspace(-1, 1, np.prod(shape), dtype=cp.float32).reshape(shape)
    compiled = fuse(score)
    expected = score(x)
    actual = compiled(x)
    assert actual.dtype == expected.dtype
    cp.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-6)
    assert compiled.diagnostics['status'] == 'executed'
    assert compiled.diagnostics['regions'][0]['backend'] in (
        'legacy_fusion', 'new_fusion')
    assert compiled.diagnostics['kernel_launches'] is None
    compiled(x)
    assert compiled.diagnostics['cache_hit']


def test_nested_functions_and_varying_scalar(gpu):
    def helper(x, scale):
        return cp.sin(x) * scale

    def root(x, scale):
        return helper(x, scale) + 1

    x = cp.arange(16, dtype=cp.float32)
    compiled = fuse(root)
    for scale in (np.float32(0.25), np.float32(0.75)):
        cp.testing.assert_allclose(compiled(x, scale), root(x, scale))
    assert len(compiled._plans) == 1
    assert len(compiled._analyzer.functions) == 2


def test_shape_branch_guards_across_same_dtype_and_ndim(gpu):
    def function(x):
        if x.shape[0] > 8:
            return cp.sin(x)
        return cp.cos(x)

    compiled = fuse(function)
    for size in (4, 16, 6, 32):
        x = cp.arange(size, dtype=cp.float32)
        cp.testing.assert_allclose(compiled(x), function(x))
    assert len(compiled._plans) == 1
    assert len(next(iter(compiled._plans.values()))) == 2


def test_allocations_are_runtime_boundaries(gpu):
    def function(x):
        y = cp.ones(x.shape, dtype=cp.float32)
        return cp.sin(x + y)

    compiled = fuse(function)
    for size in (4, 12):
        x = cp.arange(size, dtype=cp.float32)
        cp.testing.assert_allclose(compiled(x), function(x))
    assert len(compiled.diagnostics['allocations']) == 1
    assert compiled.diagnostics['cache_hit']


def test_instance_attributes_rebound_and_methods_inline(gpu):
    class Model:
        def __init__(self, scale):
            self.scale = scale

        def helper(self, x):
            return x * self.scale

        @fuse
        def apply(self, x):
            return self.helper(x) + 1

    x = cp.arange(8, dtype=cp.float32)
    first, second = Model(np.float32(2)), Model(np.float32(3))
    for model in (first, second):
        cp.testing.assert_allclose(model.apply(x), x * model.scale + 1)
    first.scale = np.float32(4)
    cp.testing.assert_allclose(first.apply(x), x * 4 + 1)
    assert len(Model.apply._analyzer.functions) == 2
    assert len(Model.apply._plans) == 1


def test_tuple_outputs_and_strides(gpu):
    def function(x):
        y = cp.sin(x)
        return y + 1, y * 2

    x = cp.arange(32, dtype=cp.float32)[::-2]
    for actual, expected in zip(fuse(function)(x), function(x)):
        cp.testing.assert_allclose(actual, expected)


def test_execution_failure_is_not_replayed(gpu):
    def function(x):
        return cp.ones_like(x) + x

    x = cp.arange(8, dtype=cp.float32)
    compiled = fuse(function)
    compiled.explain(x)
    plan = next(iter(compiled._plans.values()))[0]
    # The executor must propagate a failure after the preceding allocation
    # exactly once, without restarting the original function.
    original = plan.execute.__globals__
    key = next(k for k, v in original.items()
               if v is plan.regions[0].fusion)
    failing = mock.Mock(side_effect=RuntimeError('launch failed'))
    original[key] = failing
    with pytest.raises(RuntimeError, match='launch failed'):
        compiled(x)
    failing.assert_called_once()
    assert compiled.diagnostics['status'] == 'execution_failed'


def test_new_fusion_fallback_and_state_restoration(gpu):
    from cupy._core import _fusion_thread_local

    def function(x, y):
        return cp.sin(x + y)

    compiled = fuse(function)
    x = cp.arange(4, dtype=cp.float32).reshape(4, 1)
    y = cp.arange(3, dtype=cp.float32).reshape(1, 3)
    cp.testing.assert_allclose(compiled(x, y), function(x, y))
    state = _fusion_thread_local.thread_local
    assert not state.is_old_fusing
    assert not getattr(state, 'is_new_fusing', False)


def test_explicit_numpy_scalar_dtypes_and_python_scalars(gpu):
    def function(x, scale):
        return cp.sin(x) * scale

    x = cp.arange(8, dtype=cp.float32)
    compiled = fuse(function)
    for scale in (0.5, np.float32(0.5), np.float64(0.5), 2):
        actual, expected = compiled(x, scale), function(x, scale)
        assert actual.dtype == expected.dtype
        cp.testing.assert_allclose(actual, expected)


def test_device_predicates_rejected_without_readback(gpu):
    def function(x):
        if cp.sum(x) > 0:
            return x + 1
        return x - 1

    x = cp.arange(4, dtype=cp.float32)
    with pytest.raises(Unsupported, match='Device-data predicates'):
        fuse(function).explain(x)


def test_python_not_is_not_silently_elementwise(gpu):
    def function(x):
        return not x

    x = cp.arange(4, dtype=cp.float32)
    with pytest.raises(Unsupported, match='array truth is not elementwise'):
        fuse(function).explain(x)


def test_conditional_allocation_and_changing_boolean(gpu):
    def function(x, flag):
        if flag:
            x = cp.ones_like(x)
        return cp.sin(x)

    x = cp.arange(8, dtype=cp.float32)
    compiled = fuse(function)
    assert not compiled.explain(x, False)['allocations']
    assert len(compiled.explain(x, True)['allocations']) == 1
    for flag in (True, False, True):
        cp.testing.assert_allclose(compiled(x, flag), function(x, flag))


def test_numpy_arithmetic_warnings_are_not_emulated(gpu):
    def function(a, b):
        return a / b

    # Host NumPy arithmetic would raise under this errstate. The compiled
    # operation follows CUDA division and does not inspect device results.
    with np.errstate(divide='raise', invalid='raise'):
        compiled = fuse(function)
        compiled.explain(np.float32(1), np.float32(0))
        result = compiled(np.float32(1), np.float32(0))
    assert bool(cp.isinf(result))


def test_dtype_predicate_and_method_override(gpu):
    class Model:
        def helper(self, x):
            return x + 1

        @fuse
        def apply(self, x):
            if x.dtype == cp.float32:
                return self.helper(x)
            return cp.cos(x)

    class Other(Model):
        def helper(self, x):
            return x * 2

    for dtype in (cp.float32, cp.float64):
        x = cp.arange(4, dtype=dtype)
        for model in (Model(), Other()):
            expected = model.helper(x) if dtype == cp.float32 else cp.cos(x)
            cp.testing.assert_allclose(model.apply(x), expected)
    functions = Model.apply._analyzer.functions
    assert id(Model.helper) in functions
    assert id(Other.helper) in functions


@pytest.mark.parametrize('predicate', [
    lambda x: x.dtype.kind == 'f',
    lambda x: 'f' != x.dtype.kind,
    lambda x: x.dtype.kind in 'fc',
    lambda x: x.dtype.kind not in 'fc',
    lambda x: x.dtype.kind in ('i', 'u'),
    lambda x: x.dtype.kind not in ('i', 'u'),
    lambda x: x.dtype.kind in {'i', 'u'},
    lambda x: x.dtype.kind not in {'i', 'u'},
])
def test_dtype_kind_predicates(gpu, predicate):
    def function(x):
        if predicate(x):
            return cp.not_equal(x, x)
        return cp.equal(x, x)

    compiled = fuse(function)
    for dtype in ('bool', 'int32', 'uint64', 'float32',
                  'float64', 'complex64'):
        x = cp.arange(7).astype(dtype)
        for _ in range(2):
            cp.testing.assert_array_equal(compiled(x), function(x))
            assert compiled.diagnostics['guards']
            assert '.dtype.kind' in compiled.diagnostics['guards'][0][
                'expression']
        assert compiled.diagnostics['cache_hit']


def test_dtype_kind_stored_attribute_can_change(gpu):
    class Model:
        @fuse
        def apply(self):
            dtype = self.values.dtype
            if dtype.kind in ('f', 'c'):
                return cp.not_equal(self.values, self.values)
            return cp.equal(self.values, self.values)

    model = Model()
    for dtype in ('float32', 'int32', 'float64', 'uint64', 'complex64'):
        model.values = cp.arange(7).astype(dtype)
        expected = (cp.not_equal(model.values, model.values)
                    if model.values.dtype.kind in 'fc'
                    else cp.equal(model.values, model.values))
        cp.testing.assert_array_equal(model.apply(), expected)


def test_dtype_kind_explain_without_cuda_or_user_execution():
    class ArrayMetadata:
        ndim = 1

        def __init__(self, dtype):
            self.dtype = np.dtype(dtype)

    calls = []

    def effect(x):
        calls.append(x)
        return x

    def function(x):
        if x.dtype.kind in 'iu':
            return x + 1
        return effect(x)

    compiled = fuse(function)
    with mock.patch.object(cp, 'ndarray', ArrayMetadata):
        with mock.patch.object(cp, 'empty', side_effect=AssertionError):
            report = compiled.explain(ArrayMetadata('int32'))
            assert report['guards'][0]['outcome']
            with pytest.raises(Unsupported):
                compiled.explain(ArrayMetadata('float32'))
    assert calls == []


def test_shape_membership_guards(gpu):
    def function(x):
        if x.shape[0] in (1, 4):
            return x + 1
        return x - 1

    compiled = fuse(function)
    for size in (1, 8, 4, 3):
        x = cp.arange(size, dtype=cp.float64)
        cp.testing.assert_array_equal(compiled(x), function(x))
    assert len(next(iter(compiled._plans.values()))) == 2


@pytest.mark.parametrize('tuple_container', [False, True])
def test_metadata_membership_rejects_custom_container(tuple_container):
    class Container:
        def __contains__(self, value):
            raise AssertionError('user membership executed')

        def __eq__(self, value):
            raise AssertionError('user equality executed')

    container = (Container(),) if tuple_container else Container()

    def function(x):
        if x.dtype.kind in container:
            return x + 1
        return x - 1

    class ArrayMetadata:
        dtype = np.dtype('float32')
        ndim = 1

    with mock.patch.object(cp, 'ndarray', ArrayMetadata):
        with pytest.raises(Unsupported, match='Unsupported host predicate'):
            fuse(function).explain(ArrayMetadata())


def test_dynamic_reduction_options_rejected_during_analysis():
    def function(x, axis):
        return cp.sum(x, axis=axis)

    with pytest.raises(Unsupported, match='Reduction options must be static'):
        fuse(function).explain(1, 0)


def test_shadowed_instance_dictionary_does_not_execute_property():
    class Model:
        def __init__(self):
            self.scale = 2

        @property
        def __dict__(self):
            raise AssertionError('dictionary property executed')

        @fuse
        def apply(self, x):
            return x * self.scale

    with pytest.raises(Unsupported, match='No stored instance attribute'):
        Model.apply.explain(Model(), 1)
