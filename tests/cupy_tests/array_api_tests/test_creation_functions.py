from numpy.testing import assert_raises, assert_equal
import cupy as cp
# due to the module structure we can't import it from cupy.array_api._typing
from cupy.cuda import Device

from cupy.array_api import all
from cupy.array_api._creation_functions import (
    asarray,
    arange,
    empty,
    empty_like,
    eye,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from cupy.array_api._dtypes import float32, float64
from cupy.array_api._array_object import Array


def test_asarray_errors():
    # Test various protections against incorrect usage
    assert_raises(TypeError, lambda: Array([1]))
#    assert_raises(TypeError, lambda: asarray(["a"]))  # TODO(leofang): fix this?
    assert_raises(ValueError, lambda: asarray([1.0], dtype=cp.float16))
    assert_raises(OverflowError, lambda: asarray(2**100))
    # Preferably this would be OverflowError
    # assert_raises(OverflowError, lambda: asarray([2**100]))
#    assert_raises(TypeError, lambda: asarray([2**100]))  # TODO(leofang): fix this?
    assert_raises(ValueError, lambda: asarray([1], device="cpu"))  # numpy's pick
    assert_raises(ValueError, lambda: asarray([1], device="gpu"))

    assert_raises(ValueError, lambda: asarray([1], dtype=int))
    assert_raises(ValueError, lambda: asarray([1], dtype="i"))
    asarray([1], device=Device())  # on current device


def test_asarray_copy():
    a = asarray([1])
    b = asarray(a, copy=True)
    a[0] = 0
    assert all(b[0] == 1)
    assert all(a[0] == 0)
    # Once copy=False is implemented, replace this with
    # a = asarray([1])
    # b = asarray(a, copy=False)
    # a[0] = 0
    # assert all(b[0] == 0)
    assert_raises(NotImplementedError, lambda: asarray(a, copy=False))


def test_asarray_nested():
    a = asarray([[ones(5), ones(5)], [ones(5), ones(5)]])
    assert_equal(a.shape, (2, 2, 5))


def test_arange_errors():
    assert_raises(ValueError, lambda: arange(1, device="cpu"))  # numpy's pick
    assert_raises(ValueError, lambda: arange(1, device="gpu"))
    assert_raises(ValueError, lambda: arange(1, dtype=int))
    assert_raises(ValueError, lambda: arange(1, dtype="i"))
    arange(1, device=Device())  # on current device


def test_empty_errors():
    assert_raises(ValueError, lambda: empty((1,), device="cpu"))  # numpy's pick
    assert_raises(ValueError, lambda: empty((1,), device="gpu"))
    assert_raises(ValueError, lambda: empty((1,), dtype=int))
    assert_raises(ValueError, lambda: empty((1,), dtype="i"))
    empty((1,), device=Device())  # on current device


def test_empty_like_errors():
    assert_raises(ValueError, lambda: empty_like(asarray(1), device="cpu"))  # numpy's pick
    assert_raises(ValueError, lambda: empty_like(asarray(1), device="gpu"))
    assert_raises(ValueError, lambda: empty_like(asarray(1), dtype=int))
    assert_raises(ValueError, lambda: empty_like(asarray(1), dtype="i"))
    empty_like(asarray(1), device=Device())  # on current device


def test_eye_errors():
    assert_raises(ValueError, lambda: eye(1, device="cpu"))  # numpy's pick
    assert_raises(ValueError, lambda: eye(1, device="gpu"))
    assert_raises(ValueError, lambda: eye(1, dtype=int))
    assert_raises(ValueError, lambda: eye(1, dtype="i"))
    eye(1, device=Device())  # on current device


def test_full_errors():
    assert_raises(ValueError, lambda: full((1,), 0, device="cpu"))  # numpy's pick
    assert_raises(ValueError, lambda: full((1,), 0, device="gpu"))
    assert_raises(ValueError, lambda: full((1,), 0, dtype=int))
    assert_raises(ValueError, lambda: full((1,), 0, dtype="i"))
    full((1,), 0, device=Device())  # on current device


def test_full_like_errors():
    assert_raises(ValueError, lambda: full_like(asarray(1), 0, device="cpu"))  # numpy's pick
    assert_raises(ValueError, lambda: full_like(asarray(1), 0, device="gpu"))
    assert_raises(ValueError, lambda: full_like(asarray(1), 0, dtype=int))
    assert_raises(ValueError, lambda: full_like(asarray(1), 0, dtype="i"))
    full_like(asarray(1), 0, device=Device())  # on current device


def test_linspace_errors():
    assert_raises(ValueError, lambda: linspace(0, 1, 10, device="cpu"))  # numpy's pick
    assert_raises(ValueError, lambda: linspace(0, 1, 10, device="gpu"))
    assert_raises(ValueError, lambda: linspace(0, 1, 10, dtype=float))
    assert_raises(ValueError, lambda: linspace(0, 1, 10, dtype="f"))
    linspace(0, 1, 10, device=Device())  # on current device


def test_ones_errors():
    assert_raises(ValueError, lambda: ones((1,), device="cpu"))  # numpy's pick
    assert_raises(ValueError, lambda: ones((1,), device="gpu"))
    assert_raises(ValueError, lambda: ones((1,), dtype=int))
    assert_raises(ValueError, lambda: ones((1,), dtype="i"))
    ones((1,), device=Device())  # on current device


def test_ones_like_errors():
    assert_raises(ValueError, lambda: ones_like((1,), device="cpu"))  # numpy's pick
    assert_raises(ValueError, lambda: ones_like(asarray(1), device="gpu"))
    assert_raises(ValueError, lambda: ones_like(asarray(1), dtype=int))
    assert_raises(ValueError, lambda: ones_like(asarray(1), dtype="i"))
    ones_like(asarray(1), device=Device())  # on current device


def test_zeros_errors():
    assert_raises(ValueError, lambda: zeros((1,), device="cpu"))  # numpy's pick
    assert_raises(ValueError, lambda: zeros((1,), device="gpu"))
    assert_raises(ValueError, lambda: zeros((1,), dtype=int))
    assert_raises(ValueError, lambda: zeros((1,), dtype="i"))
    zeros((1,), device=Device())  # on current device


def test_zeros_like_errors():
    assert_raises(ValueError, lambda: zeros_like((1,), device="cpu"))  # numpy's pick
    assert_raises(ValueError, lambda: zeros_like(asarray(1), device="gpu"))
    assert_raises(ValueError, lambda: zeros_like(asarray(1), dtype=int))
    assert_raises(ValueError, lambda: zeros_like(asarray(1), dtype="i"))
    zeros_like(asarray(1), device=Device())  # on current device

def test_meshgrid_dtype_errors():
    # Doesn't raise
    meshgrid()
    meshgrid(asarray([1.], dtype=float32))
    meshgrid(asarray([1.], dtype=float32), asarray([1.], dtype=float32))

    assert_raises(ValueError, lambda: meshgrid(asarray([1.], dtype=float32), asarray([1.], dtype=float64)))
