from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import cuda
from cupy import testing


# name -> make(xp, src, **kw). ``src`` is the input of the *_like and
# from_data functions; the others ignore it. ``kw`` is only passed to CuPy.
_CASES = {
    'empty': lambda xp, src, **kw: xp.empty((2, 3), **kw),
    'empty_like': lambda xp, src, **kw: xp.empty_like(src, **kw),
    'eye': lambda xp, src, **kw: xp.eye(3, 4, k=1, **kw),
    'identity': lambda xp, src, **kw: xp.identity(3, **kw),
    'ones': lambda xp, src, **kw: xp.ones((2, 3), **kw),
    'ones_like': lambda xp, src, **kw: xp.ones_like(src, **kw),
    'zeros': lambda xp, src, **kw: xp.zeros((2, 3), **kw),
    'zeros_like': lambda xp, src, **kw: xp.zeros_like(src, **kw),
    'full': lambda xp, src, **kw: xp.full((2, 3), 7, **kw),
    'full_like': lambda xp, src, **kw: xp.full_like(src, 7, **kw),
    'array': lambda xp, src, **kw: xp.array(src, **kw),
    'asarray': lambda xp, src, **kw: xp.asarray(src, **kw),
    'asanyarray': lambda xp, src, **kw: xp.asanyarray(src, **kw),
    'arange': lambda xp, src, **kw: xp.arange(1, 10, 2, **kw),
    'linspace': lambda xp, src, **kw: xp.linspace(0, 1, 5, **kw),
}

# The same functions with non-default arguments, which the ``device=`` code
# path has to pass on unchanged.
_CASES_WITH_ARGS = {
    'empty': lambda xp, src, **kw: xp.empty((2, 3), 'f', 'F', **kw),
    'empty_like': lambda xp, src, **kw: xp.empty_like(
        src, dtype='f', order='C', shape=(3, 2), **kw),
    'eye': lambda xp, src, **kw: xp.eye(2, 3, 1, 'f', 'F', **kw),
    'identity': lambda xp, src, **kw: xp.identity(3, 'f', **kw),
    'ones': lambda xp, src, **kw: xp.ones((2, 3), 'f', 'F', **kw),
    'ones_like': lambda xp, src, **kw: xp.ones_like(
        src, dtype='f', order='C', shape=(3, 2), **kw),
    'zeros': lambda xp, src, **kw: xp.zeros((2, 3), 'f', 'F', **kw),
    'zeros_like': lambda xp, src, **kw: xp.zeros_like(
        src, dtype='f', order='C', shape=(3, 2), **kw),
    'full': lambda xp, src, **kw: xp.full((2, 3), 7, 'f', 'F', **kw),
    'full_like': lambda xp, src, **kw: xp.full_like(
        src, 7, dtype='f', order='C', shape=(3, 2), **kw),
    'array': lambda xp, src, **kw: xp.array(
        src, dtype='f', order='C', ndmin=3, **kw),
    'asarray': lambda xp, src, **kw: xp.asarray(src, 'f', 'C', **kw),
    'asanyarray': lambda xp, src, **kw: xp.asanyarray(src, 'f', 'C', **kw),
    'arange': lambda xp, src, **kw: xp.arange(1, 10, 2, 'f', **kw),
    'linspace': lambda xp, src, **kw: xp.linspace(
        0, [1, 2], 5, False, dtype='f', axis=1, **kw),
}

_ALL = tuple(_CASES)
_LIKE = ('empty_like', 'ones_like', 'zeros_like', 'full_like')
_FROM_DATA = ('array', 'asarray', 'asanyarray')
_UNINITIALIZED = ('empty', 'empty_like')


def _host_src():
    return numpy.arange(6, dtype=numpy.float64).reshape(2, 3)


_HOST_INPUTS = [
    pytest.param(lambda: _host_src(), id='numpy'),
    pytest.param(lambda: _host_src().tolist(), id='list'),
]


def _create(name, src=None, cases=_CASES, **kw):
    """Returns (CuPy result, expected NumPy result) for case ``name``.

    ``src`` defaults to a CuPy input on the current device.
    """
    host = _host_src() if src is None else cupy.asnumpy(src)
    if src is None:
        src = cupy.asarray(host)
    return cases[name](cupy, src, **kw), cases[name](numpy, host)


def _check(name, result, expected, device_id):
    assert isinstance(result, cupy.ndarray)
    assert result.device.id == device_id
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    if name not in _UNINITIALIZED:
        testing.assert_allclose(result, expected)


@pytest.fixture(autouse=True)
def _current_device_unchanged():
    # Creating an array must never leave another device current.
    before = cuda.runtime.getDevice()
    yield
    assert cuda.runtime.getDevice() == before


# With peer access between GPUs, an operation running on the wrong device
# only warns instead of failing, so make the warning an error.
@pytest.mark.filterwarnings('error::cupy._util.PerformanceWarning')
class TestDeviceArgument:

    @pytest.mark.parametrize('name', _ALL)
    def test_none_is_current_device(self, name):
        result, expected = _create(name, device=None)
        _check(name, result, expected, cuda.runtime.getDevice())

    @pytest.mark.parametrize('as_type', [int, cuda.Device])
    @pytest.mark.parametrize('name', _ALL)
    def test_current_device(self, name, as_type):
        dev_id = cuda.runtime.getDevice()
        result, expected = _create(name, device=as_type(dev_id))
        _check(name, result, expected, dev_id)

    @testing.multi_gpu(2)
    @pytest.mark.parametrize('as_type', [int, cuda.Device])
    @pytest.mark.parametrize('name', _ALL)
    def test_other_device(self, name, as_type):
        with cuda.Device(0):
            result, expected = _create(name, device=as_type(1))
            assert cuda.runtime.getDevice() == 0
        _check(name, result, expected, 1)

    @testing.multi_gpu(2)
    @pytest.mark.parametrize('name', _ALL)
    def test_overrides_device_context(self, name):
        with cuda.Device(1):
            result, expected = _create(name, device=0)
            assert cuda.runtime.getDevice() == 1
        _check(name, result, expected, 0)

    @testing.multi_gpu(2)
    @pytest.mark.parametrize('name', _LIKE + _FROM_DATA)
    def test_input_on_other_device(self, name):
        with cuda.Device(0):
            src = cupy.asarray(_host_src())
        result, expected = _create(name, src, device=1)
        assert result is not src
        _check(name, result, expected, 1)

    @testing.multi_gpu(2)
    @pytest.mark.parametrize('name', _LIKE)
    def test_like_without_device_uses_current_device(self, name):
        # The Array API asks for the device of the input instead; changing
        # that is left to a separate change.
        with cuda.Device(0):
            src = cupy.asarray(_host_src())
        with cuda.Device(1):
            result, expected = _create(name, src)
        _check(name, result, expected, 1)

    @testing.multi_gpu(2)
    @pytest.mark.parametrize('make_src', _HOST_INPUTS)
    @pytest.mark.parametrize('name', _FROM_DATA)
    def test_host_input(self, name, make_src):
        src = make_src()
        result = _CASES[name](cupy, src, device=1)
        _check(name, result, _CASES[name](numpy, src), 1)

    @pytest.mark.parametrize('name', ('asarray', 'asanyarray'))
    def test_same_device_input_is_not_copied(self, name):
        src = cupy.asarray(_host_src())
        assert _CASES[name](cupy, src, device=src.device) is src

    @testing.multi_gpu(2)
    @pytest.mark.parametrize('name', _ALL)
    def test_other_arguments_are_kept(self, name):
        with cuda.Device(0):
            src = cupy.asfortranarray(cupy.asarray(_host_src()))
        result, expected = _create(
            name, src, cases=_CASES_WITH_ARGS, device=1)
        _check(name, result, expected, 1)
        assert result.flags.c_contiguous == expected.flags.c_contiguous
        assert result.flags.f_contiguous == expected.flags.f_contiguous

    @testing.multi_gpu(2)
    @pytest.mark.parametrize('name', _LIKE)
    def test_like_keeps_layout_of_non_contiguous_input(self, name):
        # A non-contiguous input with order='K' makes empty_like allocate
        # with explicit strides, a separate code path.
        host = _host_src()[:, ::2].T
        with cuda.Device(0):
            src = cupy.asarray(_host_src())[:, ::2].T
        result = _CASES[name](cupy, src, device=1)
        expected = _CASES[name](numpy, host)
        _check(name, result, expected, 1)
        assert result.strides == expected.strides

    def test_empty_like_device_is_keyword_only(self):
        a = cupy.arange(3)
        with pytest.raises(TypeError):
            cupy.empty_like(a, None, 'K', None, None, 0)

    @testing.multi_gpu(2)
    def test_linspace_retstep(self):
        result, step = cupy.linspace(0, 1, 5, retstep=True, device=1)
        assert result.device.id == 1
        assert step == 0.25

    @testing.multi_gpu(2)
    def test_full_fill_value_on_other_device(self):
        with cuda.Device(0):
            value = cupy.asarray(3.0)
        result = cupy.full((4,), value, device=1)
        assert result.device.id == 1
        testing.assert_array_equal(result, numpy.full(4, 3.0))

    @pytest.mark.parametrize('bad', ['cpu', 'cuda:0', 1.0, True, False])
    @pytest.mark.parametrize('name', _ALL)
    def test_invalid_device_type(self, name, bad):
        with pytest.raises(TypeError):
            _create(name, device=bad)

    @pytest.mark.parametrize('name', _ALL)
    def test_invalid_device_id(self, name):
        n_gpus = cuda.runtime.getDeviceCount()
        with pytest.raises(cuda.runtime.CUDARuntimeError):
            _create(name, device=n_gpus)

    @testing.multi_gpu(2)
    def test_error_after_switch_restores_device(self):
        a = cupy.arange(3)
        with pytest.raises(ValueError):
            cupy.zeros(-1, device=1)
        assert cuda.runtime.getDevice() == 0
        with pytest.raises(TypeError):
            cupy.empty_like(a, subok=True, device=1)
        assert cuda.runtime.getDevice() == 0
