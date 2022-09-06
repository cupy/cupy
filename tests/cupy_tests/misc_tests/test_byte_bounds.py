import cupy
from cupy import testing


@testing.gpu
class TestByteBounds:

    @testing.for_all_dtypes()
    def test_1d_contiguous(self, dtype):
        a = cupy.zeros(12, dtype=dtype)
        itemsize = a.itemsize
        a_low = a.data.ptr
        a_high = a.data.ptr + 12 * itemsize
        assert cupy.byte_bounds(a) == (a_low, a_high)

    @testing.for_all_dtypes()
    def test_2d_contiguous(self, dtype):
        a = cupy.zeros((4, 7), dtype=dtype)
        itemsize = a.itemsize
        a_low = a.data.ptr
        a_high = a.data.ptr + 4 * 7 * itemsize
        assert cupy.byte_bounds(a) == (a_low, a_high)

    @testing.for_all_dtypes()
    def test_1d_noncontiguous_pos_stride(self, dtype):
        a = cupy.zeros(12, dtype=dtype)
        itemsize = a.itemsize
        b = a[::2]
        b_low = b.data.ptr
        b_high = b.data.ptr + 11 * itemsize  # a[10]
        assert cupy.byte_bounds(b) == (b_low, b_high)

    @testing.for_all_dtypes()
    def test_2d_noncontiguous_pos_stride(self, dtype):
        a = cupy.zeros((4, 7), dtype=dtype)
        b = a[::2, ::2]
        itemsize = b.itemsize
        b_low = a.data.ptr
        b_high = b.data.ptr + 3 * 7 * itemsize  # a[2][6]
        assert cupy.byte_bounds(b) == (b_low, b_high)

    @testing.for_all_dtypes()
    def test_1d_contiguous_neg_stride(self, dtype):
        a = cupy.zeros(12, dtype=dtype)
        b = a[::-1]
        itemsize = b.itemsize
        b_low = b.data.ptr - 11 * itemsize
        b_high = b.data.ptr + 1 * itemsize
        assert cupy.byte_bounds(b) == (b_low, b_high)

    @testing.for_all_dtypes()
    def test_2d_noncontiguous_neg_stride(self, dtype):
        a = cupy.zeros((4, 7), dtype=dtype)
        b = a[::-2, ::-2]  # strides = (-56, -8), shape = (2, 4)
        itemsize = b.itemsize
        b_low = b.data.ptr - 2 * 7 * itemsize * \
            (2 - 1) - 2 * itemsize * (4 - 1)
        b_high = b.data.ptr + 1 * itemsize
        assert cupy.byte_bounds(b) == (b_low, b_high)

    @testing.for_all_dtypes()
    def test_2d_noncontiguous_posneg_stride_1(self, dtype):
        a = cupy.zeros((4, 7), dtype=dtype)
        b = a[::1, ::-1]  # strides = (28, -4), shape=(4, 7)
        itemsize = b.itemsize
        b_low = b.data.ptr - itemsize * (7 - 1)
        b_high = b.data.ptr + 1 * itemsize + 7 * itemsize * (4 - 1)
        assert cupy.byte_bounds(b) == (b_low, b_high)

    @testing.for_all_dtypes()
    def test_2d_noncontiguous_posneg_stride_2(self, dtype):
        a = cupy.zeros((4, 7), dtype=dtype)
        b = a[::2, ::-2]  # strides = (56, -8), shape=(2, 4)
        itemsize = b.itemsize
        b_low = b.data.ptr - 2 * itemsize * (4 - 1)
        b_high = b.data.ptr + 1 * itemsize + 2 * 7 * itemsize * (2 - 1)
        assert cupy.byte_bounds(b) == (b_low, b_high)
