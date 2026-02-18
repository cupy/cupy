from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import testing
from cupy.testing._protocol_helpers import DummyObjectWithCudaArrayInterface


class TestCreation:
    def test_empty(self):
        a = cupy.empty((2, 3, 4), dtype="i,i")
        assert a.dtype == "i,i"
        assert a.shape == (2, 3, 4)

    @testing.numpy_cupy_array_equal()
    def test_zeros(self, xp):
        return xp.zeros((2, 3, 4), dtype="i,i")

    @pytest.mark.parametrize("func", [
        cupy.array,
        cupy.asarray,
        lambda x: cupy.asarray([x, x]),
    ])
    def test_reject_references(self, func):
        arr = numpy.array([1, 2, 3], dtype="q,O")
        # Check coming from a NumPy array
        with pytest.raises(ValueError):
            func(arr)

        if func is cupy.asarray:
            # We currently allow viewing into a GPU array unsafely...
            return

        # Check coming from a bad cupy array-like
        base = cupy.zeros(3, dtype="q,q")

        class ByteSwapped(DummyObjectWithCudaArrayInterface):
            @property
            def __cuda_array_interface__(self):
                # Fake the dtype to something bad
                iface = super().__cuda_array_interface__
                iface["typestr"] = arr.dtype.str
                iface["descr"] = arr.dtype.descr
                return iface

        with pytest.raises(ValueError):
            func(ByteSwapped(base))


class TestFieldAccess:
    @testing.numpy_cupy_array_equal()
    @pytest.mark.parametrize("index", ["f0", "f1"])
    def test_getitem(self, xp, index):
        a = xp.array([(1, 2), (3, 4)], dtype="i,i")
        res = a[index]
        assert a.flags.c_contiguous and a.flags.f_contiguous
        assert not res.flags.c_contiguous
        assert not res.flags.f_contiguous
        return res

    @testing.numpy_cupy_array_equal()
    @pytest.mark.parametrize("index, value", [("f0", -1), ("f1", -2)])
    def test_setitem(self, xp, index, value):
        a = xp.array([(1, 2), (3, 4)], dtype="i,i")
        a[index] = value
        return a

    @pytest.mark.parametrize("field, dtype", [("f1", "?,f8"), ("f0", "f8,?")])
    def test_bad_alignment(self, field, dtype):
        # If a field is not aligned (default for NumPy) using it leads to
        # RuntimeError. It may make sense to raise a nicer error earlier
        # (e.g. at indexing time)
        a = cupy.array([(1, 2), (3, 4), (5, 6), (7, 8)], dtype=dtype)
        msg = "result array with.* sufficiently aligned for GPU"
        with pytest.raises(ValueError, match=msg):
            a[field]

        with pytest.raises(ValueError, match=msg):
            a[field] = 1

    @testing.numpy_cupy_array_equal()
    @pytest.mark.parametrize("field, dtype", [("f1", "?,f8"), ("f0", "f8,?")])
    def test_alignment_empty_ok(self, xp, field, dtype):
        a = xp.array([(1, 2), (3, 4), (5, 6), (7, 8)], dtype=dtype)
        a = a[:0]
        return a[field]

    @testing.numpy_cupy_array_equal()
    def test_nested(self, xp):
        dtype = xp.dtype([("a", "i,i"), ("b", "i,i")])
        a = xp.array([((1, 2), (3, 4)), ((5, 6), (7, 8))], dtype=dtype)

        # Chained indexing into substructures is OK since the intermediate is.
        a["a"]["f0"] += a["b"]["f1"]
        return a

    def test_nested_subarray(self):
        # We could support it, but it changes the resulting array shape.
        a = cupy.zeros(10, dtype="i,(3,3)i")
        msg = "CuPy does not yet support accessing nested subarrays"
        with pytest.raises(ValueError, match=msg):
            a["f1"]


class TestIndexing:
    @testing.numpy_cupy_array_equal()
    @pytest.mark.parametrize("sl", [slice(1, None), slice(None, None, 2)])
    def test_slicing(self, xp, sl):
        # As of writing, we can't copy a structured array because that
        # requires a kernel launch.  But we can slice it fine.
        a = xp.array([(1, 2), (3, 4), (5, 6)], dtype="i,i")
        # Compare fields, because we cannot copy the strided structured array
        # to a contiguous one to copy back to the CPU.
        return (a[sl]["f0"], a[sl]["f1"])
