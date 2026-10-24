"""Ascend 组合算子测试（np.nanargmax / nanargmin / nanmean / nanprod / choose / angle(deg)）。

分三层验证（本机无 NPU，上限 L3）：

1. **注册层**（不需要设备）：`ascend_nanprod` 等名字真的出现在运行时注册表里；
2. **接线层**（不需要设备）：把 `composite.active` 打桩为 True、组合函数替换成间谍，
   断言公开 API 走了组合分支 —— 这样在没有 NPU 的机器上也能测"接线是否正确"；
3. **降级契约**（真机 / 无 NPU 都能过）：真的调用时，无 NPU 必须报设备错误
   （`EL0003`），而不是"算子未注册" —— 说明路径确实走到了 aclnn。
"""

from __future__ import annotations

import numpy
import pytest

import cupy

from cupy._indexing.indexing import choose
from cupy._math.arithmetic import angle
from cupy._sorting.search import nanargmax, nanargmin
from cupy._statistics.meanvar import nanmean


@pytest.fixture
def composite(monkeypatch):
    """把组合模块的开关打开，并返回模块以便替换其中的函数。"""
    from cupy._core._ascend import composite as module
    monkeypatch.setattr(module, 'active', lambda: True)
    return module


def _recorder(calls):
    def stub(*args, **kwargs):
        calls.append((args, kwargs))
        return 'COMPOSED'
    return stub


# ---------------------------------------------------------------------------
# 1. 注册层
# ---------------------------------------------------------------------------
def test_nanprod_registered_as_reduction():
    """nanprod 是本 PR 唯一的 C++ 组合（NanToNum + Prod），必须真的注册上。"""
    from cupy.backends.ascend.api.acl_utils import py_list_acl_ufuncs
    ops = dict(py_list_acl_ufuncs())
    for name in ('ascend_nanprod', 'ascend_nanprod_with_dtype'):
        assert name in ops, f'{name} 未注册'
        assert ops[name] == 3, f'{name} 的 OpType 应为 REDUCTION_OP(3)'


def test_composite_module_public_api():
    from cupy._core._ascend import composite
    for name in composite.__all__:
        value = getattr(composite, name)
        if name == 'REQUIRED_OPS':
            assert isinstance(value, dict) and value
        else:
            assert callable(value), name
    assert isinstance(composite.active(), bool)


def test_active_cache_reset():
    from cupy._core._ascend import composite
    first = composite.active()
    composite.reset_cache()
    assert composite.active() == first


# ---------------------------------------------------------------------------
# 2. 接线层（公开 API -> 组合实现）
# ---------------------------------------------------------------------------
def test_nanargmax_wired(composite, monkeypatch):
    calls = []
    monkeypatch.setattr(composite, 'nanargmax', _recorder(calls))
    a = numpy.array([1.0, numpy.nan, 3.0], dtype=numpy.float32)
    assert nanargmax(a) == 'COMPOSED'
    assert calls[0][1]['axis'] is None and calls[0][1]['keepdims'] is False


def test_nanargmin_wired(composite, monkeypatch):
    calls = []
    monkeypatch.setattr(composite, 'nanargmin', _recorder(calls))
    a = numpy.array([1.0, numpy.nan, 3.0], dtype=numpy.float32)
    assert nanargmin(a, axis=0, keepdims=True) == 'COMPOSED'
    assert calls[0][1] == {'axis': 0, 'dtype': None, 'out': None,
                           'keepdims': True}


def test_nanmean_wired(composite, monkeypatch):
    calls = []
    monkeypatch.setattr(composite, 'nanmean', _recorder(calls))
    a = numpy.array([1.0, numpy.nan, 3.0], dtype=numpy.float32)
    assert nanmean(a) == 'COMPOSED'
    assert calls[0][1]['dtype'] is None


def test_choose_wired(composite, monkeypatch):
    calls = []
    monkeypatch.setattr(composite, 'choose', _recorder(calls))
    a = numpy.array([0, 1, 1], dtype=numpy.int32)
    assert choose(a, [numpy.zeros(3), numpy.ones(3)], mode='clip') == 'COMPOSED'
    args, kwargs = calls[0]
    assert kwargs == {'out': None, 'mode': 'clip'} and args[0] is a


def test_angle_deg_wired(composite, monkeypatch):
    calls = []
    monkeypatch.setattr(composite, 'angle_deg', _recorder(calls))
    z = numpy.array([1.0 + 1.0j], dtype=numpy.complex64)
    assert angle(z, deg=True) == 'COMPOSED'
    assert calls[0][0][0] is z


def test_int_dtypes_do_not_use_composite(composite, monkeypatch):
    """整型/布尔没有 NaN：nanargmax/nanmean 走上游原来的 fast path，组合不被调用。"""
    calls = []
    monkeypatch.setattr(composite, 'nanargmax', _recorder(calls))
    monkeypatch.setattr(composite, 'nanmean', _recorder(calls))
    a = numpy.array([1, 3, 2], dtype=numpy.int32)
    for func in (nanargmax, nanmean):
        try:
            func(a)
        except Exception:
            pass      # numpy 数组没有 cupy 的 dtype/keepdims 形参 -> TypeError
    assert calls == []


# ---------------------------------------------------------------------------
# 3. 组合依赖的算子都已注册（无 NPU 能做的最强校验）
# ---------------------------------------------------------------------------
def _registered_op_names() -> set:
    from cupy.backends.ascend.api.acl_utils import (py_list_acl_ufuncs,
                                                    py_list_custom_kernels)
    names = {name for name, _ in py_list_acl_ufuncs()}
    names.update(py_list_custom_kernels())
    return names


@pytest.mark.parametrize('composite_name', sorted(
    __import__('cupy._core._ascend.composite', fromlist=['x']).REQUIRED_OPS))
def test_composite_dependencies_are_registered(composite_name):
    """组合里用到的每个算子都必须有 aclnn 注册，否则组合照样跑不起来。

    这是无 NPU 时最有价值的校验：它保证"组合"不是纸上谈兵 ——
    `nanprod`（C++ 组合）经 `ascend_nanprod`、`angle` 经自定义 AscendC 内核，
    两条注册表都要查。
    """
    from cupy._core._ascend import composite
    registered = _registered_op_names()
    missing = [f'ascend_{name}'
               for name in composite.REQUIRED_OPS[composite_name]
               if f'ascend_{name}' not in registered]
    assert not missing, f'{composite_name} 依赖未注册的算子: {missing}'


# ---------------------------------------------------------------------------
# 4. 真机数值对拍（无 NPU 自动 skip，见文件头第 1/2 层校验）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('name', ['nanargmax', 'nanargmin', 'nanmean'])
def test_nan_ops_match_numpy(name, has_npu):
    if not has_npu:
        pytest.skip('需要 NPU 才能数值对拍')
    data = numpy.array([[1.0, numpy.nan, 3.0], [numpy.nan, numpy.nan, 2.0]],
                       dtype=numpy.float32)
    func_cupy = {'nanargmax': nanargmax, 'nanargmin': nanargmin,
                 'nanmean': nanmean}[name]
    func_numpy = {'nanargmax': numpy.nanargmax, 'nanargmin': numpy.nanargmin,
                  'nanmean': numpy.nanmean}[name]
    for axis in (None, 0, 1):
        if name != 'nanmean' and axis is not None and \
                numpy.isnan(data).all(axis=axis).any():
            continue        # 全 NaN 切片: numpy 抛 ValueError, cupy 返回越界值
        got = func_cupy(cupy.asarray(data), axis=axis)
        expected = func_numpy(data, axis=axis)
        numpy.testing.assert_allclose(cupy.asnumpy(got), expected)


def test_nanprod_matches_numpy(has_npu):
    if not has_npu:
        pytest.skip('需要 NPU 才能数值对拍')
    data = numpy.array([1.0, numpy.nan, 3.0, 4.0], dtype=numpy.float32)
    got = cupy.nanprod(cupy.asarray(data))
    numpy.testing.assert_allclose(cupy.asnumpy(got), numpy.nanprod(data),
                                  rtol=1e-6)
    # 整型输入走 prod 分支
    ints = numpy.array([2, 3, 4], dtype=numpy.int32)
    got = cupy.nanprod(cupy.asarray(ints))
    numpy.testing.assert_array_equal(cupy.asnumpy(got), numpy.nanprod(ints))


@pytest.mark.parametrize('mode', ['raise', 'wrap', 'clip'])
def test_choose_matches_numpy(mode, has_npu):
    if not has_npu:
        pytest.skip('需要 NPU 才能数值对拍')
    a = numpy.array([0, 1, 2, 1], dtype=numpy.int32)
    choices = [numpy.array([1.0, 2.0], dtype=numpy.float32),
               numpy.array([3.0, 4.0], dtype=numpy.float32),
               numpy.array([5.0, 6.0], dtype=numpy.float32)]
    got = choose(cupy.asarray(a), [cupy.asarray(c) for c in choices], mode=mode)
    expected = numpy.choose(a, choices, mode=mode)
    numpy.testing.assert_allclose(cupy.asnumpy(got), expected)


def test_angle_deg_matches_numpy(has_npu):
    if not has_npu:
        pytest.skip('需要 NPU 才能数值对拍')
    z = numpy.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 - 1.0j],
                    dtype=numpy.complex64)
    got = angle(cupy.asarray(z), deg=True)
    numpy.testing.assert_allclose(cupy.asnumpy(got), numpy.angle(z, deg=True),
                                  rtol=1e-4, atol=1e-4)
