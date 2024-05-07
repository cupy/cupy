# coding: utf-8

import numpy
import operator
import pytest

import cupy
from cupy import testing


class TestStrings:
    @testing.for_dtypes(["S", "U", "S10", "U10"])
    def test_roundtrip_numpy(self, dtype):
        a = numpy.array(["spam", "cushion", "parrot", "pirate"], dtype=dtype)
        c = cupy.asarray(a)
        assert c.dtype == a.dtype
        testing.assert_array_equal(c.get(), a)

    @testing.for_dtypes(["S", "U5"])
    @testing.numpy_cupy_array_equal()
    def test_string_cast(self, xp, dtype):
        # Cupy supports the cast (although it can't raise an error on unicode)
        a = xp.array(["spam", "cushion", "parrot", "pirate"], dtype=dtype)
        to_dtype = "U" if dtype == "S" else "S"
        return a.astype(to_dtype)

    @testing.for_dtypes(["S4", "U5"])
    @pytest.mark.parametrize("cmp", ["eq", "ne", "lt", "le", "gt", "ge"])
    @testing.numpy_cupy_array_equal()
    def test_string_comparisons_simple(self, xp, dtype, cmp):
        a = xp.array(["0", "", "9", "10", "100", "2", "1000"], dtype=dtype)
        b = xp.array(["10"], dtype=dtype)
        op = getattr(operator, cmp)
        return op(a, b)

    @testing.numpy_cupy_array_equal()
    def test_unicode_comparisons_unicode(self, xp):
        a = xp.array(["Ú", "🦜", "早"])
        return a == xp.array(["🦜"])

    @testing.for_int_dtypes(no_bool=True)
    def test_integer_string_casts(self, dtype):
        iinfo = numpy.iinfo(dtype)
        # NOTE: The minimum value currently fails
        int_list = [0, iinfo.max, iinfo.min+1, 100]
        string_list = [str(_) for _ in list(int_list)]
        bytes_list = [_.encode() for _ in list(string_list)]

        ints = cupy.array(int_list, dtype=dtype)
        assert list(ints.get()) == int_list
        assert list(ints.astype("U").get()) == string_list
        assert list(ints.astype("S").get()) == bytes_list
        # also test some too short casts:
        assert list(ints.astype("U3").get()) == [s[:3] for s in string_list]
        # test round-triping back:
        assert list(ints.astype("U").astype(dtype).get()) == int_list
        assert list(ints.astype("S").astype(dtype).get()) == int_list

    @testing.for_dtypes("SU")
    def test_indexing(self, dtype):
        ints = cupy.arange(1000)
        strings = ints.astype(dtype)
        strings = strings[[3, 6, 100, -1]]
        expected = cupy.array(["3", "6", "100", "999"], dtype=strings.dtype)
        testing.assert_array_equal(strings, expected)
