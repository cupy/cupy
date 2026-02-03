from __future__ import annotations

import operator

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

        # Right now, structured dtypes round-trip as V<itemsize> and lose
        # their structure in the __cuda_array_interface__. As long as that
        # is the case, this always passes.
        # with pytest.raises(ValueError):
        #     func(ByteSwapped(base))
        if func != cupy.asarray:
            with pytest.raises(TypeError):
                func(ByteSwapped(base))
        else:
            assert func(ByteSwapped(base)).dtype == f"V{base.dtype.itemsize}"


class TestFieldCasting:
    @testing.numpy_cupy_array_equal()
    def test_casting_simple(self, xp):
        a = xp.array([(1, 2.), (3, 4.)], dtype="i8,f8")
        return a.astype("f8,i8")

    @testing.numpy_cupy_array_equal()
    def test_casting_swapped_fields(self, xp):
        a = xp.array([(1, 2.), (3, 4.)], dtype="i8,f8")
        return a[["f0", "f1"]].astype("f8,i8")

    @testing.numpy_cupy_array_equal()
    def test_casting_nested(self, xp):
        dt_nested = xp.dtype("i8,f8")
        dtype = xp.dtype([("a", "f8"), ("b", dt_nested)])
        a = xp.array([(1, 2.), (3, 4.)], dtype=dtype)

        new_nested = xp.dtype("f4,i4")
        new_dtype = xp.dtype([("a", "f4"), ("b", new_nested)])
        return a.astype(new_dtype)


@pytest.mark.parametrize("op", [operator.eq, operator.ne])
class TestComparison:
    @testing.numpy_cupy_array_equal()
    def test_simple(self, xp, op):
        a = xp.array([(1, 2)], dtype="i8,f8")
        b = xp.array([(1, 2.), (1, 3), (3, 2)], dtype="i8,f8")
        return op(a, b)

    @testing.numpy_cupy_array_equal()
    def test_with_cast_simple(self, xp, op):
        a = xp.array([(1, 2)], dtype="i8,f8")
        b = xp.array([(1, 2.), (1, 3), (3, 2)], dtype="f8,i8")
        return op(a, b)

    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_field_order(self, xp, op):
        # Swapped fields with name mis-match are unclear how they
        # should be compared, so NumPy raises.
        a = xp.array([(1, 2)], dtype="i8,f8")[["f1", "f0"]]
        b = xp.array([(1, 2.), (1, 3), (3, 2)], dtype="f8,i8")
        return op(a, b)

    @testing.numpy_cupy_array_equal()
    def test_nested(self, xp, op):
        adt_nested = xp.dtype("i8,f8")
        adtype = xp.dtype([("a", "f8"), ("b", adt_nested)])
        bdt_nested = xp.dtype("f4,i4")
        bdtype = xp.dtype([("a", "f4"), ("b", bdt_nested)])

        a = xp.array([(1, (2, 3))], dtype=adtype)
        b = xp.array([(1, (2, 3)), (1, (2, 4)), (1, (3, 3))], dtype=bdtype)

        return op(a, b)

    def test_promotion_aligned(self, op):
        a = cupy.array([(1, 2)], dtype="f8,c8")
        b = cupy.array([(1, 2), (1, 3), (2, 3)], dtype="f8,f8")
        # For comparison, promotion would go to f8,c16
        cmp_dtype = numpy.result_type(a.dtype, b.dtype)
        assert cmp_dtype.itemsize == 24
        # However, this isn't aligned for the GPU which wants itemsize
        # alignment for complex.
        assert cupy.make_gpu_aligned_dtype(cmp_dtype).itemsize == 32
        res = op(a, b)
        assert numpy.array_equal(res.get(), op(a.get(), b.get()))


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

    @testing.numpy_cupy_array_equal()
    def test_multiple_fields(self, xp):
        a = xp.ones(10, "i,f,i")
        a["f1"] = xp.arange(10)
        a["f2"] = -xp.arange(10)
        return a[["f2", "f1"]]

    @testing.numpy_cupy_array_equal()
    def test_multifield_assignment(self, xp):
        a = xp.ones(10, "i,f,i,f,i,f")
        for i in range(6):
            a[f'f{i}'] = xp.arange(i, 10+i)

        # in-place modify `a` with mixed fields that have holes.
        # This also requires casting the fields.
        # NOTE(seberg): Behavior is not identical if source and destination
        # point to the same memory. Don't do that... (i.e. the copy is needed)
        a[["f1", "f0", "f4"]] = a[["f2", "f1", "f5"]].copy()
        return a


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
