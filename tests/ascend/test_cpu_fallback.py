"""Ascend CPU fallback 测试（det / eigvals / eigh / cholesky … 搬到 host 用 NumPy）。

分三层验证（本机无 NPU，上限 L3）：

1. **基础设施层**（不需要设备）：注册表与 ``numpy.linalg`` 一一对应、
   D2H/H2D 的搬运语义、``DISABLE_ENV`` 开关、未注册名字响亮失败；
   H2D 用假的 ``cupy`` 模块替身，因此不需要真的 NPU。
2. **接线层**（不需要设备）：把 ``is_ascend`` 打桩为 True、``cpu_fallback.call``
   换成间谍，断言 ``cupy.linalg.det`` 等公开 API 真的走了 fallback 分支，
   且传下去的名字与 :data:`FALLBACKS` 的 key 一致（名字写错只会静默不生效）。
   另有源码级校验覆盖无法用 NumPy 数组触发的 ``cholesky``（它要求真的 cupy 数组）。
3. **真机数值对拍**（无 NPU 自动 skip）。
"""

from __future__ import annotations

import collections
import inspect
import os

import numpy
import pytest

import cupy
from cupy._core._ascend import cpu_fallback


EXPECTED_FALLBACKS = {
    'linalg.cholesky': 'cholesky',
    'linalg.det': 'det',
    'linalg.slogdet': 'slogdet',
    'linalg.eig': 'eig',
    'linalg.eigvals': 'eigvals',
    'linalg.eigh': 'eigh',
    'linalg.eigvalsh': 'eigvalsh',
}


@pytest.fixture
def no_device_cupy(monkeypatch):
    """把 ``cpu_fallback._cupy()`` 换成一个不让真设备参与的替身。

    真实 D2H/H2D 需要 NPU；这里只验证"哪些值会被搬、搬成什么"。
    ``asarray`` 直接返回入参，于是 ``to_device`` 的结果就是 NumPy 对象。
    """
    fake = type('FakeCupy', (), {'ndarray': (), 'asarray': staticmethod(lambda x: x)})
    monkeypatch.setattr(cpu_fallback, '_cupy', lambda: fake)
    return fake


@pytest.fixture
def fallback_active(monkeypatch):
    """打开 fallback 开关（绕过真实后端判断）。"""
    monkeypatch.setattr(cpu_fallback, '_ASCEND', True)
    monkeypatch.delenv(cpu_fallback.DISABLE_ENV, raising=False)
    return cpu_fallback


# ---------------------------------------------------------------------------
# 1. 基础设施层
# ---------------------------------------------------------------------------
def test_public_api():
    for name in cpu_fallback.__all__:
        value = getattr(cpu_fallback, name)
        if name == 'FALLBACKS':
            assert isinstance(value, dict) and value
        elif name == 'DISABLE_ENV':
            assert isinstance(value, str)
        else:
            assert callable(value), name
    assert isinstance(cpu_fallback.active(), bool)
    assert isinstance(cpu_fallback.available('linalg.det'), bool)


def test_registry_matches_numpy_linalg():
    """每个注册项都必须指向真的 ``numpy.linalg`` 实现（防止名字写错/指向自己）。"""
    assert set(cpu_fallback.FALLBACKS) == set(EXPECTED_FALLBACKS)
    for key, attr in EXPECTED_FALLBACKS.items():
        assert cpu_fallback.FALLBACKS[key] is getattr(numpy.linalg, attr), key


def test_active_cache_reset():
    first = cpu_fallback.active()
    cpu_fallback.reset_cache()
    assert cpu_fallback.active() == first


def test_disable_env_switch(monkeypatch):
    monkeypatch.setattr(cpu_fallback, '_ASCEND', True)
    monkeypatch.delenv(cpu_fallback.DISABLE_ENV, raising=False)
    assert cpu_fallback.active() is True
    for value in ('1', 'true', 'True'):
        monkeypatch.setenv(cpu_fallback.DISABLE_ENV, value)
        assert cpu_fallback.active() is False, value
    monkeypatch.setenv(cpu_fallback.DISABLE_ENV, '0')
    assert cpu_fallback.active() is True


def test_call_raises_when_disabled(monkeypatch, no_device_cupy):
    monkeypatch.setattr(cpu_fallback, '_ASCEND', True)
    monkeypatch.setenv(cpu_fallback.DISABLE_ENV, '1')
    with pytest.raises(NotImplementedError, match='CPU fallback 未启用'):
        cpu_fallback.call('linalg.det', numpy.eye(2))


def test_call_unknown_name(fallback_active, no_device_cupy):
    with pytest.raises(KeyError, match='未知的 CPU fallback 算子'):
        cpu_fallback.call('linalg.does_not_exist', numpy.eye(2))


def test_to_numpy_passthrough(no_device_cupy):
    """非设备数组原样返回（不拷贝、不转换）。"""
    arr = numpy.arange(3)
    assert cpu_fallback.to_numpy(arr) is arr
    assert cpu_fallback.to_numpy(3) == 3
    assert cpu_fallback.to_numpy(None) is None


def test_to_device_passthrough(no_device_cupy):
    assert cpu_fallback.to_device(3) == 3
    assert cpu_fallback.to_device('f') == 'f'
    assert cpu_fallback.to_device(None) is None
    assert cpu_fallback.to_device((1, 2)) == (1, 2)
    assert cpu_fallback.to_device([1, 2]) == [1, 2]


def test_to_device_converts_numpy(no_device_cupy):
    arr = numpy.zeros(3, dtype=numpy.float32)
    assert cpu_fallback.to_device(arr) is arr          # fake asarray 直接返回
    assert cpu_fallback.to_device(numpy.float32(1.5)) == numpy.float32(1.5)


def test_to_device_keeps_namedtuple_structure(no_device_cupy):
    """``numpy.linalg`` 返回 namedtuple（如 SlogdetResult），字段要被搬运。"""
    Pair = collections.namedtuple('Pair', ['a', 'b'])
    out = cpu_fallback.to_device(Pair(numpy.zeros(1), numpy.ones(1)))
    assert isinstance(out, Pair)
    assert out.a is not None and out.b is not None


def test_run_computes_with_host_numpy(fallback_active, no_device_cupy):
    """``run`` 的端到端语义：host 上算完再搬回（这里用替身观察结果）。"""
    a = numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32)
    assert cpu_fallback.run(numpy.linalg.det, a) == pytest.approx(-2.0)
    assert cpu_fallback.call('linalg.det', a) == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# 2. 接线层（cupy.linalg.* -> cpu_fallback.call）
# ---------------------------------------------------------------------------
def _install_ascend(monkeypatch):
    """让 ``is_ascend`` 属性在所有调用点都为 True。"""
    import cupy.backends.backend as backend
    monkeypatch.setattr(backend, 'is_ascend', True)


def _install_call_spy(monkeypatch, calls):
    def spy(name, *args, **kwargs):
        calls.append((name, args, kwargs))
        return 'FALLBACK'
    monkeypatch.setattr(cpu_fallback, 'call', spy)


WIRING_CASES = [
    ('det', (numpy.eye(2, dtype=numpy.float32),), {}),
    ('slogdet', (numpy.eye(2, dtype=numpy.float32),), {}),
    ('eig', (numpy.eye(2, dtype=numpy.float32),), {}),
    ('eigvals', (numpy.eye(2, dtype=numpy.float32),), {}),
    ('eigh', (numpy.eye(2, dtype=numpy.float32),), {}),
    ('eigvalsh', (numpy.eye(2, dtype=numpy.float32),), {}),
]


@pytest.mark.parametrize('name, args, kwargs', WIRING_CASES)
def test_linalg_wired_to_fallback(monkeypatch, name, args, kwargs):
    _install_ascend(monkeypatch)
    calls = []
    _install_call_spy(monkeypatch, calls)
    func = getattr(cupy.linalg, name)
    assert func(*args, **kwargs) == 'FALLBACK'
    assert calls[0][0] == 'linalg.' + name
    assert cpu_fallback.available(calls[0][0]), '接线用的名字必须在注册表里'


@pytest.mark.parametrize('name', ['eigh', 'eigvalsh'])
def test_uplo_forwarded(monkeypatch, name):
    """``UPLO`` 必须透传给 NumPy（L/U 影响只读哪半边矩阵）。"""
    _install_ascend(monkeypatch)
    calls = []
    _install_call_spy(monkeypatch, calls)
    getattr(cupy.linalg, name)(numpy.eye(2, dtype=numpy.float32), 'U')
    assert calls[0][2] == {'UPLO': 'U'}


@pytest.mark.parametrize('name', sorted(EXPECTED_FALLBACKS))
def test_wiring_name_appears_in_source(name):
    """源码级校验：函数体里确实调用了 ``cpu_fallback.call('<name>')``。

    覆盖 ``cholesky``（它要求真的 cupy 数组，无法用 NumPy 数组触发），
    同时能捉住"注册名拼错 -> 静默不生效"这类问题。
    """
    func = getattr(cupy.linalg, name.split('.')[1])
    source = inspect.getsource(func)
    assert 'cpu_fallback.call({!r}'.format(name) in source, source
    assert 'is_ascend' in source


def test_cholesky_still_rejects_non_cupy_array(monkeypatch):
    """fallback 分支不能绕过 ``_assert_cupy_array``（错误要先于设备拷贝抛出）。"""
    _install_ascend(monkeypatch)
    calls = []
    _install_call_spy(monkeypatch, calls)
    with pytest.raises(numpy.linalg.LinAlgError):
        cupy.linalg.cholesky(numpy.eye(2, dtype=numpy.float32))
    assert calls == []


def test_cuda_path_untouched(monkeypatch):
    """``is_ascend`` 为 False 时不能碰 fallback（CUDA 后端仍有 cuSOLVER 实现）。"""
    import cupy.backends.backend as backend
    monkeypatch.setattr(backend, 'is_ascend', False)
    calls = []
    _install_call_spy(monkeypatch, calls)
    for name in ('det', 'eigvals', 'eigh'):
        try:
            getattr(cupy.linalg, name)(numpy.eye(2))
        except Exception:
            pass          # 无 CUDA 库/无设备时抛错无妨，关键是没走 fallback
    assert calls == []


# ---------------------------------------------------------------------------
# 3. 真机数值对拍
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('dtype', [numpy.float32, numpy.float64])
def test_det_matches_numpy(dtype, has_npu):
    if not has_npu:
        pytest.skip('需要 NPU 才能数值对拍')
    data = numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    got = cupy.linalg.det(cupy.asarray(data))
    numpy.testing.assert_allclose(cupy.asnumpy(got), numpy.linalg.det(data),
                                  rtol=1e-5)


def test_eigvals_matches_numpy(has_npu):
    if not has_npu:
        pytest.skip('需要 NPU 才能数值对拍')
    data = numpy.array([[0.0, -1.0], [1.0, 0.0]], dtype=numpy.float32)
    got = cupy.linalg.eigvals(cupy.asarray(data))
    got, expected = cupy.asnumpy(got), numpy.linalg.eigvals(data)
    numpy.testing.assert_allclose(
        numpy.sort_complex(got), numpy.sort_complex(expected), rtol=1e-5)


def test_eigh_matches_numpy(has_npu):
    if not has_npu:
        pytest.skip('需要 NPU 才能数值对拍')
    data = numpy.array([[2.0, 1.0], [1.0, 2.0]], dtype=numpy.float32)
    got_w, got_v = cupy.linalg.eigh(cupy.asarray(data))
    exp_w, exp_v = numpy.linalg.eigh(data)
    numpy.testing.assert_allclose(cupy.asnumpy(got_w), exp_w, rtol=1e-5)
    numpy.testing.assert_allclose(numpy.abs(cupy.asnumpy(got_v)),
                                  numpy.abs(exp_v), rtol=1e-5, atol=1e-5)


def test_eigvalsh_matches_numpy(has_npu):
    if not has_npu:
        pytest.skip('需要 NPU 才能数值对拍')
    data = numpy.array([[2.0, 1.0], [1.0, 2.0]], dtype=numpy.float32)
    got = cupy.linalg.eigvalsh(cupy.asarray(data))
    numpy.testing.assert_allclose(
        cupy.asnumpy(got), numpy.linalg.eigvalsh(data), rtol=1e-5)


def test_cholesky_matches_numpy(has_npu):
    if not has_npu:
        pytest.skip('需要 NPU 才能数值对拍')
    data = numpy.array([[4.0, 2.0], [2.0, 3.0]], dtype=numpy.float32)
    got = cupy.linalg.cholesky(cupy.asarray(data))
    numpy.testing.assert_allclose(
        cupy.asnumpy(got), numpy.linalg.cholesky(data), rtol=1e-5)


def test_slogdet_matches_numpy(has_npu):
    if not has_npu:
        pytest.skip('需要 NPU 才能数值对拍')
    data = numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32)
    sign, logdet = cupy.linalg.slogdet(cupy.asarray(data))
    exp_sign, exp_logdet = numpy.linalg.slogdet(data)
    assert isinstance(sign, cupy.ndarray) and isinstance(logdet, cupy.ndarray)
    numpy.testing.assert_allclose(cupy.asnumpy(sign), exp_sign)
    numpy.testing.assert_allclose(cupy.asnumpy(logdet), exp_logdet, rtol=1e-5)


def test_disable_env_makes_linalg_fail_loudly(monkeypatch, has_npu):
    """关了 fallback 就必须响亮失败（而不是悄悄走 cuSOLVER 的残骸）。"""
    if not has_npu:
        pytest.skip('需要 NPU 才能验证真实调用路径')
    monkeypatch.setenv(cpu_fallback.DISABLE_ENV, '1')
    with pytest.raises(NotImplementedError):
        cupy.linalg.det(cupy.eye(2, dtype=numpy.float32))
    assert os.environ.get(cpu_fallback.DISABLE_ENV) == '1'
