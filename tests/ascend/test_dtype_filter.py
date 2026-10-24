"""Ascend dtype 过滤策略的单元测试 (纯 Python, 不需要 NPU)。

被测对象: `cupy/testing/_ascend_dtypes.py` + `cupy.testing._loops` 的接入,
以及 `tests/conftest.py` 依赖的 `item_dtypes()` 提取规则。
"""

from __future__ import annotations

import importlib
import os

import numpy
import pytest

from cupy.testing import _ascend_dtypes as ad

# 本模块测试过滤策略本身, 需要真的使用 float64/complex 等被过滤的 dtype
pytestmark = pytest.mark.ascend_dtype_filter_off


@pytest.fixture(autouse=True)
def _reset_policy_cache():
    ad.reset_cache()
    yield
    ad.reset_cache()


class _Item:
    """最小化的 pytest item 替身: 只需要 callspec.params。"""

    def __init__(self, **params):
        self.callspec = type('Callspec', (), {'params': params})()


# ---------------------------------------------------------------------------
# 默认策略
# ---------------------------------------------------------------------------
def test_default_skip_set():
    assert ad.DEFAULT_SKIP_DTYPES == ('float64', 'complex64', 'complex128')


def test_policy_off_keeps_every_dtype():
    policy = ad.policy('off')
    assert not policy.enabled
    assert 'off' in policy.describe()
    dtypes = (numpy.float32, numpy.float64, numpy.complex128, numpy.int32)
    assert policy.filter(dtypes) == dtypes
    assert not policy.is_skipped(numpy.float64)
    assert policy.to_drop(dtypes) == []


def test_policy_on_filters_unsupported():
    policy = ad.policy('on')
    assert policy.enabled
    dtypes = (numpy.float16, numpy.float32, numpy.float64,
              numpy.complex64, numpy.complex128, numpy.int32)
    assert policy.filter(dtypes) == (numpy.float16, numpy.float32,
                                     numpy.int32)
    assert policy.to_drop(dtypes) == [numpy.float64, numpy.complex64,
                                      numpy.complex128]
    assert policy.is_skipped(numpy.float64)
    assert policy.is_skipped('complex128')
    assert not policy.is_skipped(numpy.int64)   # int64 四则运算仍可用
    reason = policy.skip_reason([numpy.complex128])
    assert 'complex128' in reason and '--ascend-dtype-filter=off' in reason


def test_env_overrides_skip_set(monkeypatch):
    monkeypatch.setenv(ad._ENV_SKIP, 'float64')
    policy = ad.policy('on')
    assert policy.names == ('float64',)
    assert policy.is_skipped(numpy.float64)
    assert not policy.is_skipped(numpy.complex128)


def test_env_can_disable_filter(monkeypatch):
    monkeypatch.setenv(ad._ENV_FILTER, 'off')
    assert not ad.policy().enabled


def test_auto_mode_follows_backend(monkeypatch):
    monkeypatch.delenv(ad._ENV_FILTER, raising=False)
    monkeypatch.setattr(ad, 'is_ascend', lambda: True)
    assert ad.policy().enabled
    monkeypatch.setattr(ad, 'is_ascend', lambda: False)
    assert not ad.policy().enabled


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        ad.configure('sometimes')


# ---------------------------------------------------------------------------
# dtype 解析
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('value,expected', [
    (numpy.float64, 'float64'),
    (numpy.dtype('f8'), 'float64'),
    ('float32', 'float32'),
    ('complex128', 'complex128'),
    (numpy.array([1.0]).dtype, 'float64'),
])
def test_dtype_of_parses(value, expected):
    parsed = ad.dtype_of(value)
    assert parsed is not None and parsed.name == expected


@pytest.mark.parametrize('value', [None, False, 3, 'F', 'nonsense', [1, 2]])
def test_dtype_of_ignores_non_dtypes(value):
    assert ad.dtype_of(value) is None


# ---------------------------------------------------------------------------
# pytest item -> dtype 提取 (收集期 skip 的依据)
# ---------------------------------------------------------------------------
def test_item_dtypes_from_dtype_values():
    item = _Item(dtype=[numpy.float32, numpy.float64])
    assert [d.name for d in ad.item_dtypes(item)] == ['float32', 'float64']
    assert ad.policy('on').to_drop(ad.item_dtypes(item)) == \
        [numpy.dtype('float64')]


def test_item_dtypes_from_strings_with_dtype_param_name():
    assert [d.name for d in ad.item_dtypes(_Item(dtype='float64'))] == \
        ['float64']
    assert [d.name for d in ad.item_dtypes(_Item(dtyp='complex128'))] == \
        ['complex128']
    assert [d.name for d in ad.item_dtypes(_Item(dtypes=['int64']))] == \
        ['int64']


def test_item_dtypes_ignores_unrelated_params():
    # `order='F'` 不是 dtype (F 会被 numpy 解析成 complex64)
    assert ad.item_dtypes(_Item(order=['C', 'F'])) == []
    # 参数名含 float64, 但值是 bool
    assert ad.item_dtypes(_Item(float64_distances=[False, True])) == []
    # 参数名完全不像 dtype -> 字符串值不当 dtype
    assert ad.item_dtypes(_Item(kind='float64')) == []
    # 参数名像 dtype -> 即使是 "xxx_dtype_str" 也算 (确实在参数化 dtype)
    assert [d.name for d in ad.item_dtypes(_Item(out_dtype_str='float64'))] == \
        ['float64']


def test_item_dtypes_without_callspec():
    assert ad.item_dtypes(object()) == []


def test_item_dtypes_from_dynamic_class_name():
    """`cupy.testing._parameterized.product()` 生成的类把参数写进类名。"""
    nodeid = ('tests/cupy_tests/math_tests/test_arithmetic.py::'
              'ArithmeticBinaryBase_param_2389_{arg1=False, arg2=True, '
              "dtype=float64, name='floor_divide', use_dtype=False}"
              '::test_binary')
    assert [d.name for d in ad.item_dtypes_from_name(nodeid)] == ['float64']
    item = type('Item', (), {'nodeid': nodeid})()
    assert [d.name for d in ad.item_dtypes(item)] == ['float64']
    assert ad.policy('on').to_drop(ad.item_dtypes(item)) == \
        [numpy.dtype('float64')]


@pytest.mark.parametrize('name,expected', [
    # float32 也会被解析出来, 只是不在跳过集合里 (不会被 skip)
    ('X_param_1_{dtype=float32}', ['float32']),
    ('X_param_1_{dtype=complex128}', ['complex128']),
    ("X_param_1_{dtype='float64'}", []),          # 带引号 -> 不解析
    ('X_param_1_{use_dtype=False, name=abc}', []),
    ('X_param_1_{typecode=float64}', ['float64']),
    ('test_x[order=F]', []),                       # 排布顺序, 不是 dtype
    ('test_x[float64_distances=False]', []),
])
def test_item_dtypes_from_name_cases(name, expected):
    assert [d.name for d in ad.item_dtypes_from_name(name)] == expected


# ---------------------------------------------------------------------------
# 与 cupy.testing._loops 的接入
# ---------------------------------------------------------------------------
def _reload_loops():
    import cupy.testing._loops as loops
    return importlib.reload(loops)


def test_loops_candidates_filtered_when_enabled():
    ad.configure('on')
    loops = _reload_loops()
    try:
        assert numpy.float64 not in loops._regular_float_dtypes
        assert numpy.complex128 not in loops._complex_dtypes
        assert loops._complex_dtypes == ()
        assert numpy.float32 in loops._regular_float_dtypes
        # for_all_dtypes() 的组合里也不应出现 float64/complex
        all_dtypes = loops._make_all_dtypes(
            no_float16=True, no_bool=False, no_complex=False)
        names = {numpy.dtype(d).name for d in all_dtypes}
        assert 'float64' not in names and 'complex128' not in names
        assert {'float32', 'int32', 'bool'} <= names
    finally:
        ad.configure('off')
        _reload_loops()
        ad.reset_cache()


def test_loops_keeps_upstream_list_when_disabled():
    ad.configure('off')
    loops = _reload_loops()
    try:
        assert loops._regular_float_dtypes == (numpy.float64, numpy.float32)
        assert loops._complex_dtypes == (numpy.complex64, numpy.complex128)
    finally:
        ad.configure(None)
        _reload_loops()


def test_enabled_matches_policy():
    for mode in ('on', 'off'):
        ad.configure(mode)
        assert ad.enabled() == ad.policy().enabled
    ad.configure(None)


def test_describe_mentions_mode():
    ad.configure('off')
    assert 'off' in ad.describe()
    ad.configure('on')
    assert 'on' in ad.describe() and 'float64' in ad.describe()
    ad.configure(None)


def test_conftest_helper_is_importable():
    """tests/conftest.py 里的钩子依赖这些名字, 防止被改坏。"""
    for name in ('policy', 'item_dtypes', 'describe'):
        assert callable(getattr(ad, name))
    assert os.environ.get('CUPY_TEST_ASCEND_DTYPE_FILTER') is None or \
        isinstance(os.environ['CUPY_TEST_ASCEND_DTYPE_FILTER'], str)
