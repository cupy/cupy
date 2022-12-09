import numpy

import pytest

import cupy
from cupy import testing
from cupy_backends.cuda.api import runtime
from cupyx import jit


class TestThrust:
    def test_count_shared_memory(self):
        @jit.rawkernel()
        def count(x, y):
            tid = jit.threadIdx.x
            smem = jit.shared_memory(numpy.int32, 32)
            smem[tid] = x[tid]
            jit.syncthreads()
            y[tid] = jit.thrust.count(jit.thrust.device, smem, smem + 32, tid)

        size = cupy.uint32(32)
        x = cupy.arange(size, dtype=cupy.int32)
        y = cupy.zeros(size, dtype=cupy.int32)
        count[1, 32](x, y)
        testing.assert_array_equal(y, cupy.ones(size, dtype=cupy.int32))

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_adjacent_difference(self, order):
        @jit.rawkernel()
        def adjacent_difference(x, y):
            i = jit.threadIdx.x
            array = x[i]
            result = y[i]
            jit.thrust.adjacent_difference(
                jit.thrust.device, array.begin(), array.end(), result.begin())

        h, w = (128, 128)
        x = testing.shaped_random(
            (h, w), dtype=numpy.int32, order=order)
        y = cupy.zeros(h * w, dtype=cupy.int32).reshape(h, w)
        adjacent_difference[1, 128](x, y)
        testing.assert_array_equal(y[:, 0], x[:, 0])
        testing.assert_array_equal(y[:, 1:], x[:, 1:] - x[:, :-1])

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_binary_search(self, order):
        @jit.rawkernel()
        def binary_search(x, y):
            i = jit.threadIdx.x
            array = x[i]
            y[i] = jit.thrust.binary_search(
                jit.thrust.seq, array.begin(), array.end(), 100)

        n1, n2 = (128, 160)
        x = testing.shaped_random(
            (n1, n2), dtype=numpy.int32, scale=200, order=order)
        x = cupy.sort(x, axis=-1)
        y = cupy.zeros(n1, dtype=cupy.bool_)
        binary_search[1, n1](x, y)

        expected = (x == 100).any(axis=-1)
        assert 70 < expected.sum().item() < 80
        testing.assert_array_equal(y, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_binary_search_vec(self, order):
        @jit.rawkernel()
        def binary_search(x, y, z):
            i = jit.threadIdx.x
            array = x[i]
            value = y[i]
            output = z[i]
            jit.thrust.binary_search(
                jit.thrust.seq,
                array.begin(), array.end(),
                value.begin(), value.end(),
                output.begin())

        n1, n2, n3 = (128, 160, 200)
        x = testing.shaped_random(
            (n1, n2), dtype=numpy.int32, scale=200, order=order, seed=0)
        x = cupy.sort(x, axis=-1)
        y = testing.shaped_random(
            (n1, n3), dtype=numpy.int32, scale=200, order=order, seed=1)
        z = cupy.zeros((n1, n3), dtype=cupy.bool_)
        binary_search[1, n1](x, y, z)

        expected = (x[:, :, None] == y[:, None, :]).any(axis=1)
        testing.assert_array_equal(z, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_count_iterator(self, order):
        @jit.rawkernel()
        def count(x, y):
            i = jit.threadIdx.x
            j = jit.threadIdx.y
            k = jit.threadIdx.z
            array = x[i, j]
            y[i, j, k] = jit.thrust.count(
                jit.thrust.device, array.begin(), array.end(), k)

        h, w, c = (16, 16, 128)
        x = testing.shaped_random(
            (h, w, c), dtype=numpy.int32, scale=4, order=order)
        y = cupy.zeros(h * w * 4, dtype=cupy.int32).reshape(h, w, 4)
        count[1, (16, 16, 4)](x, y)
        expected = (x[..., None] == cupy.arange(4)).sum(axis=2)
        testing.assert_array_equal(y, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_copy_iterator(self, order):
        @jit.rawkernel()
        def copy(x, y):
            i = jit.threadIdx.x
            x_array = x[i]
            y_array = y[i]
            jit.thrust.copy(
                jit.thrust.device,
                x_array.begin(),
                x_array.end(),
                y_array.begin(),
            )

        h, w = (256, 256)
        x = testing.shaped_random((h, w), dtype=numpy.int32, order=order)
        y = cupy.zeros((h, w), dtype=numpy.int32)
        copy[1, 256](x, y)
        testing.assert_array_equal(x, y)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_equal(self, order):
        if runtime.is_hip:
            # TODO(asi1024): Fix compile error.
            pytest.skip('Skip rocm thrust::equal test')

        @jit.rawkernel()
        def equal(x, y, z):
            i = jit.threadIdx.x
            x_array = x[i]
            y_array = y[i]
            z[i] = jit.thrust.equal(
                jit.thrust.device,
                x_array.begin(),
                x_array.end(),
                y_array.begin(),
            )

        n1, n2 = (256, 256)
        x = testing.shaped_random(
            (n1, n2), dtype=numpy.int32, order=order, seed=0)
        y = x.copy()
        y[100][200] = 0
        z = cupy.zeros((n1,), dtype=numpy.int32)
        equal[1, 256](x, y, z)
        testing.assert_array_equal(z, (x == y).all(axis=1))

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_equal_range(self, order):
        @jit.rawkernel()
        def equal_range(x, y):
            i = jit.threadIdx.x
            array = x[i]
            start, end = jit.thrust.equal_range(
                jit.thrust.seq, array.begin(), array.end(), 100)
            y[i] = end - start

        n1, n2 = (128, 160)
        x = testing.shaped_random(
            (n1, n2), dtype=numpy.int32, scale=200, order=order)
        x = cupy.sort(x, axis=-1)
        y = cupy.zeros(n1, dtype=cupy.int32)
        equal_range[1, n1](x, y)

        expected = (x == 100).sum(axis=-1)
        testing.assert_array_equal(y, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_exclusive_scan(self, order):
        @jit.rawkernel()
        def exclusive_scan(x, y):
            i = jit.threadIdx.x
            array = x[i]
            result = y[i]
            jit.thrust.exclusive_scan(
                jit.thrust.seq, array.begin(), array.end(), result.begin())

        n1, n2 = (128, 160)
        x = testing.shaped_random(
            (n1, n2), dtype=numpy.int32, order=order)
        y = cupy.zeros((n1, n2), dtype=cupy.int32)
        exclusive_scan[1, n1](x, y)

        expected = x.cumsum(axis=-1)[:, :-1]
        testing.assert_array_equal(y[:, 1:], expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_exclusive_scan_init(self, order):
        @jit.rawkernel()
        def exclusive_scan_init(x, y):
            i = jit.threadIdx.x
            array = x[i]
            result = y[i]
            jit.thrust.exclusive_scan(
                jit.thrust.seq, array.begin(), array.end(), result.begin(), 10)

        n1, n2 = (128, 160)
        x = testing.shaped_random(
            (n1, n2), dtype=numpy.int32, order=order)
        y = cupy.zeros((n1, n2), dtype=cupy.int32)
        exclusive_scan_init[1, n1](x, y)

        expected = x.cumsum(axis=-1)[:, :-1] + 10
        testing.assert_array_equal(y[:, 1:], expected)

    def test_exclusive_scan_by_key(self):
        @jit.rawkernel()
        def exclusive_scan_by_key(key, value):
            jit.thrust.exclusive_scan_by_key(
                jit.thrust.device, key.begin(), key.end(),
                value.begin(), value.begin(), 5)

        key = cupy.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
        value = cupy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        exclusive_scan_by_key[1, 1](key, value)

        expected = cupy.array([5, 6, 7, 5, 6, 5, 5, 6, 7, 8])
        testing.assert_array_equal(value, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_fill(self, order):
        @jit.rawkernel()
        def fill(x):
            i = jit.threadIdx.x
            array = x[i]
            jit.thrust.fill(jit.thrust.device, array.begin(), array.end(), 10)

        n1, n2 = (128, 160)
        x = cupy.zeros((n1, n2), dtype=numpy.int32, order=order)
        fill[1, n1](x)
        expected = cupy.full((n1, n2), 10, dtype=numpy.int32)
        testing.assert_array_equal(x, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_find(self, order):
        @jit.rawkernel()
        def find(x, y):
            i, j, k = jit.threadIdx.x, jit.threadIdx.y, jit.threadIdx.z
            array = x[i, j]
            iterator = jit.thrust.find(
                jit.thrust.device, array.begin(), array.end(), k)
            y[i, j, k] = iterator - array.begin()

        h, w, c = (16, 16, 128)
        x = testing.shaped_random(
            (h, w, c), dtype=numpy.int32, scale=4, order=order)
        y = cupy.zeros(h * w * 4, dtype=cupy.int32).reshape(h, w, 4)
        find[1, (16, 16, 4)](x, y)

        expected = numpy.full((h, w, 4), c, y.dtype)
        x_numpy = cupy.asnumpy(x)
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    elem = x_numpy[i, j, k]
                    expected[i, j, elem] = min(expected[i, j, elem], k)
        testing.assert_array_equal(y, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_gather(self, order):
        @jit.rawkernel()
        def gather(x, y, z):
            i = jit.threadIdx.x
            map_ = x[i]
            input_ = y
            result = z[i]
            jit.thrust.gather(
                jit.thrust.device, map_.begin(), map_.end(),
                input_.begin(), result.begin())

        n1, n2 = (128, 160)
        x = testing.shaped_random(
            (n1, n2), dtype=cupy.int32, scale=n2, order=order, seed=0)
        y = testing.shaped_random(
            (n2,), dtype=cupy.float32, order=order, seed=1)
        z = cupy.zeros((n1, n2), cupy.float32)
        gather[1, n1](x, y, z)
        expected = y[x]
        testing.assert_array_equal(z, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_inclusive_scan(self, order):
        @jit.rawkernel()
        def inclusive_scan(x, y):
            i = jit.threadIdx.x
            array = x[i]
            result = y[i]
            jit.thrust.inclusive_scan(
                jit.thrust.seq, array.begin(), array.end(), result.begin())

        n1, n2 = (128, 160)
        x = testing.shaped_random(
            (n1, n2), dtype=numpy.int32, order=order)
        y = cupy.zeros((n1, n2), dtype=cupy.int32)
        inclusive_scan[1, n1](x, y)

        expected = x.cumsum(axis=-1)
        testing.assert_array_equal(y, expected)

    def test_inclusive_scan_by_key(self):
        @jit.rawkernel()
        def inclusive_scan_by_key(key, value):
            jit.thrust.inclusive_scan_by_key(
                jit.thrust.device, key.begin(), key.end(),
                value.begin(), value.begin())

        key = cupy.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
        value = cupy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        inclusive_scan_by_key[1, 1](key, value)

        expected = cupy.array([1, 2, 3, 1, 2, 1, 1, 2, 3, 4])
        testing.assert_array_equal(value, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_inner_product(self, order):
        @jit.rawkernel()
        def inner_product(a, b, c):
            i = jit.threadIdx.x
            c[i] = jit.thrust.inner_product(
                jit.thrust.device, a[i].begin(), a[i].end(),
                b[i].begin(), 0.)

        (n1, n2) = (128, 160)
        x = testing.shaped_random(
            (n1, n2), dtype=numpy.float32, order=order, seed=0)
        y = testing.shaped_random(
            (n1, n2), dtype=numpy.float32, order=order, seed=1)
        z = cupy.zeros((n1,), dtype=numpy.float32)
        inner_product[1, n1](x, y, z)

        expected = (x * y).sum(axis=1)
        testing.assert_allclose(z, expected, rtol=1e-6)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_is_sorted(self, order):
        @jit.rawkernel()
        def is_sorted(x, out):
            i = jit.threadIdx.x
            out[i] = jit.thrust.is_sorted(
                jit.thrust.device, x[i].begin(), x[i].end())

        x = cupy.array([
            [1, 4, 2, 8, 5, 7],
            [1, 2, 4, 5, 7, 8],
        ], order=order)
        out = cupy.array([False, False], numpy.bool_)
        is_sorted[1, 2](x, out)

        expected = cupy.array([False, True])
        testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_is_sorted_until(self, order):
        @jit.rawkernel()
        def is_sorted_until(x, out):
            i = jit.threadIdx.x
            it = jit.thrust.is_sorted_until(
                jit.thrust.device, x[i].begin(), x[i].end())
            out[i] = it - x[i].begin()

        x = cupy.array([
            [1, 2, 3, 4, 1, 2, 3, 4],
            [1, 2, 4, 1, 2, 3, 5, 5],
        ], order=order)
        out = cupy.zeros((2,), dtype=numpy.int64)
        is_sorted_until[1, 2](x, out)

        expected = cupy.array([4, 3])
        testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_lower_bound(self, order):
        @jit.rawkernel()
        def lower_bound(x, v, out):
            i = jit.threadIdx.x
            it = jit.thrust.lower_bound(
                jit.thrust.seq, x.begin(), x.end(), v[i])
            out[i] = it - x.begin()

        n1, n2 = (128, 160)
        x = testing.shaped_random((n2,), dtype=numpy.int32, order=order)
        x = cupy.sort(x)
        values = testing.shaped_random((n1,), dtype=numpy.int32, order=order)
        out = cupy.zeros(n1, dtype=numpy.int32)
        lower_bound[1, n1](x, values, out)

        expected = cupy.searchsorted(x, values, side='left')
        testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_lower_bound_vec(self, order):
        @jit.rawkernel()
        def lower_bound(x, v, out):
            i = jit.threadIdx.x
            jit.thrust.lower_bound(
                jit.thrust.seq, x.begin(), x.end(),
                v[i].begin(), v[i].end(), out[i].begin())

        n1, n2, n3 = (128, 160, 200)
        x = testing.shaped_random(
            (n2,), dtype=numpy.int32, scale=200, order=order, seed=0)
        x = cupy.sort(x)
        values = testing.shaped_random(
            (n1, n3), dtype=numpy.int32, scale=200, order=order, seed=1)
        out = cupy.zeros((n1, n3), dtype=numpy.int32)
        lower_bound[1, n1](x, values, out)

        expected = cupy.searchsorted(x, values, side='left')
        testing.assert_array_equal(out, expected)

    def test_make_constant_iterator(self):
        @jit.rawkernel()
        def make_constant_iterator(x):
            i = jit.threadIdx.x
            it = jit.thrust.make_constant_iterator(i * i)
            x[i] = it[i]

        n = 128
        out = cupy.zeros((n,), numpy.int32)
        make_constant_iterator[1, n](out)

        expected = cupy.arange(n) ** 2
        testing.assert_array_equal(out, expected)

    def test_make_counting_iterator(self):
        @jit.rawkernel()
        def make_counting_iterator(x):
            i = jit.threadIdx.x
            it = jit.thrust.make_counting_iterator(i * i)
            x[i] = it[i]

        n = 128
        out = cupy.zeros((n,), numpy.int32)
        make_counting_iterator[1, n](out)

        expected = cupy.arange(n) ** 2 + cupy.arange(n)
        testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_mismatch_iterator(self, order):
        if runtime.is_hip:
            pytest.xfail('HIP does not support pair of pointer type')

        @jit.rawkernel()
        def mismatch(x1, x2, out1, out2):
            i = jit.threadIdx.x
            x1_array = x1[i]
            x2_array = x2[i]
            pair = jit.thrust.mismatch(
                jit.thrust.device,
                x1_array.begin(),
                x1_array.end(),
                x2_array.begin(),
            )
            out1[i] = pair[0] - x1_array.begin()
            out2[i] = pair[1] - x2_array.begin()

        h, w = (5, 256)
        x1 = testing.shaped_random(
            (h, w), dtype=numpy.float32, scale=20000, order=order, seed=0)
        x2 = x1.copy()
        x2[0][0] = -1
        x2[1][100] = -1
        x2[2][30] = -1
        x2[3][200] = -1
        out1 = cupy.zeros(5, numpy.int32)
        out2 = cupy.zeros(5, numpy.int32)
        mismatch[1, 5](x1, x2, out1, out2)

        testing.assert_array_equal(out1, [0, 100, 30, 200, w])
        testing.assert_array_equal(out2, [0, 100, 30, 200, w])

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_reduce(self, order):
        @jit.rawkernel()
        def reduce(x, y):
            i = jit.threadIdx.x
            y[i] = jit.thrust.reduce(
                jit.thrust.device, x[i].begin(), x[i].end())

        (n1, n2) = (128, 160)
        x = testing.shaped_random((n1, n2), dtype=numpy.int64, order=order)
        out = cupy.zeros((n1,), numpy.int32)
        reduce[1, n1](x, out)

        expected = x.sum(axis=-1)
        testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_reduce_init(self, order):
        @jit.rawkernel()
        def reduce(x, y):
            i = jit.threadIdx.x
            y[i] = jit.thrust.reduce(
                jit.thrust.device, x[i].begin(), x[i].end(), 10)

        (n1, n2) = (128, 160)
        x = testing.shaped_random((n1, n2), dtype=numpy.int64, order=order)
        out = cupy.zeros((n1,), numpy.int32)
        reduce[1, n1](x, out)

        expected = x.sum(axis=-1) + 10
        testing.assert_array_equal(out, expected)

    def test_reduce_by_key(self):
        @jit.rawkernel()
        def reduce_by_key(keys, values, keys_out, values_out, size1, size2):
            ret = jit.thrust.reduce_by_key(
                jit.thrust.device, keys.begin(), keys.end(),
                values.begin(), keys_out.begin(), values_out.begin())
            size1[0] = ret[0] - keys_out.begin()
            size2[0] = ret[1] - values_out.begin()

        keys = cupy.array([1, 3, 3, 3, 2, 2, 1], dtype=numpy.int32)
        values = cupy.array([9, 8, 7, 6, 5, 4, 3], dtype=numpy.int32)
        keys_out = cupy.zeros((7,), dtype=numpy.int32)
        values_out = cupy.zeros((7,), dtype=numpy.int32)
        size1 = cupy.zeros((1,), numpy.int32)
        size2 = cupy.zeros((2,), numpy.int32)
        reduce_by_key[1, 1](keys, values, keys_out, values_out, size1, size2)

        testing.assert_array_equal(size1[0], 4)
        testing.assert_array_equal(size2[0], 4)
        testing.assert_array_equal(
            keys_out[:4], cupy.array([1, 3, 2, 1], dtype=numpy.int32))
        testing.assert_array_equal(
            values_out[:4], cupy.array([9, 21, 9, 3], dtype=numpy.int32))

    def test_remove(self):
        @jit.rawkernel()
        def remove_(x, size):
            ptr = jit.thrust.remove(
                jit.thrust.device, x.begin(), x.end(), 1)
            size[0] = ptr - x.begin()

        x = cupy.array([3, 1, 4, 1, 5, 9], dtype=numpy.int32)
        size = cupy.zeros((1,), numpy.int32)
        remove_[1, 1](x, size)

        testing.assert_array_equal(size[0], 4)
        testing.assert_array_equal(
            x[:4], cupy.array([3, 4, 5, 9], dtype=numpy.int32))

    def test_remove_copy(self):
        @jit.rawkernel()
        def remove_copy(x, result, size):
            ptr = jit.thrust.remove_copy(
                jit.thrust.device, x.begin(), x.end(), result.begin(), 1)
            size[0] = ptr - result.begin()

        x = cupy.array([3, 1, 4, 1, 5, 9], dtype=numpy.int32)
        result = cupy.zeros((6,), dtype=numpy.int32)
        size = cupy.zeros((1,), numpy.int32)
        remove_copy[1, 1](x, result, size)

        testing.assert_array_equal(size[0], 4)
        testing.assert_array_equal(
            result[:4], cupy.array([3, 4, 5, 9], dtype=numpy.int32))

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_replace(self, order):
        @jit.rawkernel()
        def replace(x):
            i = jit.threadIdx.x
            jit.thrust.replace(
                jit.thrust.device, x[i].begin(), x[i].end(), 0, 999)

        (n1, n2) = (128, 160)
        x = testing.shaped_random((n1, n2), dtype=numpy.int32, order=order)
        expected = x.copy()
        replace[1, n1](x)

        mask = expected == 0
        assert (mask.sum(axis=1) > 0).all()
        expected[mask] = 999
        testing.assert_array_equal(x, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_replace_copy(self, order):
        @jit.rawkernel()
        def replace_copy(x, out):
            i = jit.threadIdx.x
            jit.thrust.replace_copy(
                jit.thrust.device, x[i].begin(), x[i].end(),
                out[i].begin(), 0, 999)

        (n1, n2) = (128, 160)
        x = testing.shaped_random((n1, n2), dtype=numpy.int32, order=order)
        out = cupy.zeros((n1, n2), dtype=numpy.int32)
        replace_copy[1, n1](x, out)

        expected = x.copy()
        mask = x == 0
        assert (mask.sum(axis=1) > 0).all()
        expected[mask] = 999
        testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_reverse(self, order):
        @jit.rawkernel()
        def reverse(x):
            i = jit.threadIdx.x
            jit.thrust.reverse(
                jit.thrust.device, x[i].begin(), x[i].end())

        (n1, n2) = (128, 160)
        x = testing.shaped_random((n1, n2), dtype=numpy.int32, order=order)
        expected = x[:, ::-1].copy()
        reverse[1, n1](x)

        testing.assert_array_equal(x, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_reverse_copy(self, order):
        @jit.rawkernel()
        def reverse_copy(x, out):
            i = jit.threadIdx.x
            jit.thrust.reverse_copy(
                jit.thrust.device, x[i].begin(), x[i].end(), out[i].begin())

        (n1, n2) = (128, 160)
        x = testing.shaped_random((n1, n2), dtype=numpy.int32, order=order)
        out = cupy.zeros((n1, n2), dtype=numpy.int32)
        reverse_copy[1, n1](x, out)

        expected = x[:, ::-1]
        testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_scatter(self, order):
        @jit.rawkernel()
        def scatter(values, map, result):
            jit.thrust.scatter(
                jit.thrust.device, values.begin(), values.end(),
                map.begin(), result.begin())

        values = cupy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=numpy.int32)
        map = cupy.array([0, 5, 1, 6, 2, 7, 3, 8, 4, 9], dtype=numpy.int32)
        result = cupy.zeros((10,), dtype=numpy.int64)
        scatter[1, 1](values, map, result)

        testing.assert_array_equal(
            result,
            cupy.array([1, 3, 5, 7, 9, 2, 4, 6, 8, 10]),
        )

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_sequence(self, order):
        @jit.rawkernel()
        def sequence(x):
            i = jit.threadIdx.x
            jit.thrust.sequence(jit.thrust.device, x[i].begin(), x[i].end())

        (n1, n2) = (128, 160)
        x = cupy.zeros((n1, n2), dtype=numpy.int32)
        sequence[1, n1](x)

        expected = cupy.arange(n2, dtype=numpy.int32)
        expected = expected + cupy.zeros((n1, n2), dtype=numpy.int32)
        testing.assert_array_equal(x, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_sequence_init(self, order):
        @jit.rawkernel()
        def sequence(x):
            i = jit.threadIdx.x
            jit.thrust.sequence(jit.thrust.device, x[i].begin(), x[i].end(), 1)

        (n1, n2) = (128, 160)
        x = cupy.zeros((n1, n2), dtype=numpy.int32)
        sequence[1, n1](x)

        expected = cupy.arange(n2, dtype=numpy.int32)
        expected = expected + cupy.ones((n1, n2), dtype=numpy.int32)
        testing.assert_array_equal(x, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_sequence_step(self, order):
        @jit.rawkernel()
        def sequence(x):
            i = numpy.int32(jit.threadIdx.x)
            jit.thrust.sequence(
                jit.thrust.device, x[i].begin(), x[i].end(), i, 10)

        (n1, n2) = (128, 160)
        x = cupy.zeros((n1, n2), dtype=numpy.int32)
        sequence[1, n1](x)

        expected = cupy.arange(n2, dtype=numpy.int32) * 10
        expected = expected + cupy.arange(n1, dtype=numpy.int32)[:, None]
        testing.assert_array_equal(x, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_set_difference(self, order):
        if runtime.is_hip:
            pytest.xfail('HIP does not support thrust set operations.')

        @jit.rawkernel()
        def set_difference(x, y, out, size):
            it = jit.thrust.set_difference(
                jit.thrust.device,
                x.begin(), x.end(),
                y.begin(), y.end(),
                out.begin())
            size[0] = it - out.begin()

        x = cupy.array([0, 1, 3, 4, 5, 6, 9])
        y = cupy.array([1, 3, 5, 7, 9])
        out = cupy.zeros((10,), dtype=numpy.int64)
        size = cupy.zeros((1,), dtype=numpy.int64)
        set_difference[1, 1](x, y, out, size)

        testing.assert_array_equal(size, cupy.array([3]))
        testing.assert_array_equal(out[:3], cupy.array([0, 4, 6]))

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_set_difference_by_key(self, order):
        if runtime.is_hip:
            pytest.xfail('HIP does not support thrust set operations.')

        @jit.rawkernel()
        def set_difference_by_key(
                keys1, keys2, values1, values2, keys_out, values_out, size):
            it0, it1 = jit.thrust.set_difference_by_key(
                jit.thrust.device,
                keys1.begin(), keys1.end(),
                keys2.begin(), keys2.end(),
                values1.begin(), values2.begin(),
                keys_out.begin(), values_out.begin()
            )
            size[0] = it0 - keys_out.begin()
            size[1] = it1 - values_out.begin()

        a_keys = cupy.array([0, 1, 3, 4, 5, 6, 9])
        a_vals = cupy.array([2, 2, 2, 2, 2, 2, 2])
        b_keys = cupy.array([1, 3, 5, 7, 9])
        b_vals = cupy.array([1, 1, 1, 1, 1])
        keys_out = cupy.zeros((10,), dtype=numpy.int64)
        vals_out = cupy.zeros((10,), dtype=numpy.int64)
        size = cupy.zeros((2,), dtype=numpy.int64)
        set_difference_by_key[1, 1](
            a_keys, b_keys, a_vals, b_vals, keys_out, vals_out, size)

        testing.assert_array_equal(size, cupy.array([3, 3]))
        testing.assert_array_equal(keys_out[:3], cupy.array([0, 4, 6]))
        testing.assert_array_equal(vals_out[:3], cupy.array([2, 2, 2]))

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_set_intersection(self, order):
        if runtime.is_hip:
            pytest.xfail('HIP does not support thrust set operations.')

        @jit.rawkernel()
        def set_intersection(x, y, out, size):
            it = jit.thrust.set_intersection(
                jit.thrust.device,
                x.begin(), x.end(),
                y.begin(), y.end(),
                out.begin())
            size[0] = it - out.begin()

        x = cupy.array([1, 3, 5, 7, 9, 11])
        y = cupy.array([1, 1, 2, 3, 5, 8, 13])
        out = cupy.zeros((10,), dtype=numpy.int64)
        size = cupy.zeros((1,), dtype=numpy.int64)
        set_intersection[1, 1](x, y, out, size)

        testing.assert_array_equal(size, cupy.array([3]))
        testing.assert_array_equal(out[:3], cupy.array([1, 3, 5]))

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_set_intersection_by_key(self, order):
        if runtime.is_hip:
            pytest.xfail('HIP does not support thrust set operations.')

        @jit.rawkernel()
        def set_intersection_by_key(
                keys1, keys2, values1, keys_out, values_out, size):
            it0, it1 = jit.thrust.set_intersection_by_key(
                jit.thrust.device,
                keys1.begin(), keys1.end(),
                keys2.begin(), keys2.end(),
                values1.begin(), keys_out.begin(), values_out.begin()
            )
            size[0] = it0 - keys_out.begin()
            size[1] = it1 - values_out.begin()

        a_keys = cupy.array([1, 3, 5, 7, 9, 11])
        a_vals = cupy.array([2, 2, 2, 2, 2, 2, 2])
        b_keys = cupy.array([1, 1, 2, 3, 5, 8, 13])
        keys_out = cupy.zeros((10,), dtype=numpy.int64)
        vals_out = cupy.zeros((10,), dtype=numpy.int64)
        size = cupy.zeros((2,), dtype=numpy.int64)
        set_intersection_by_key[1, 1](
            a_keys, b_keys, a_vals, keys_out, vals_out, size)

        testing.assert_array_equal(size, cupy.array([3, 3]))
        testing.assert_array_equal(keys_out[:3], cupy.array([1, 3, 5]))
        testing.assert_array_equal(vals_out[:3], cupy.array([2, 2, 2]))

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_set_symmetric_difference(self, order):
        if runtime.is_hip:
            pytest.xfail('HIP does not support thrust set operations.')

        @jit.rawkernel()
        def set_symmetric_difference(x, y, out, size):
            it = jit.thrust.set_symmetric_difference(
                jit.thrust.device,
                x.begin(), x.end(),
                y.begin(), y.end(),
                out.begin())
            size[0] = it - out.begin()

        x = cupy.array([0, 1, 2, 4, 6, 7])
        y = cupy.array([1, 2, 5, 8])
        out = cupy.zeros((10,), dtype=numpy.int64)
        size = cupy.zeros((1,), dtype=numpy.int64)
        set_symmetric_difference[1, 1](x, y, out, size)

        testing.assert_array_equal(size, cupy.array([6]))
        testing.assert_array_equal(out[:6], cupy.array([0, 4, 5, 6, 7, 8]))

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_set_symmetric_difference_by_key(self, order):
        if runtime.is_hip:
            pytest.xfail('HIP does not support thrust set operations.')

        @jit.rawkernel()
        def set_symmetric_difference_by_key(
                keys1, keys2, values1, values2, keys_out, values_out, size):
            it0, it1 = jit.thrust.set_symmetric_difference_by_key(
                jit.thrust.device,
                keys1.begin(), keys1.end(),
                keys2.begin(), keys2.end(),
                values1.begin(), values2.begin(),
                keys_out.begin(), values_out.begin()
            )
            size[0] = it0 - keys_out.begin()
            size[1] = it1 - values_out.begin()

        a_keys = cupy.array([0, 1, 2, 4, 6, 7])
        a_vals = cupy.array([1, 1, 1, 1, 1, 1])
        b_keys = cupy.array([1, 2, 5, 8])
        b_vals = cupy.array([2, 2, 2, 2])
        keys_out = cupy.zeros((10,), dtype=numpy.int64)
        vals_out = cupy.zeros((10,), dtype=numpy.int64)
        size = cupy.zeros((2,), dtype=numpy.int64)
        set_symmetric_difference_by_key[1, 1](
            a_keys, b_keys, a_vals, b_vals, keys_out, vals_out, size)

        testing.assert_array_equal(size, cupy.array([6, 6]))
        testing.assert_array_equal(
            keys_out[:6], cupy.array([0, 4, 5, 6, 7, 8]))
        testing.assert_array_equal(
            vals_out[:6], cupy.array([1, 1, 2, 1, 1, 2]))

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_set_union(self, order):
        if runtime.is_hip:
            pytest.xfail('HIP does not support thrust set operations.')

        @jit.rawkernel()
        def set_union(x, y, out, size):
            it = jit.thrust.set_union(
                jit.thrust.device,
                x.begin(), x.end(),
                y.begin(), y.end(),
                out.begin())
            size[0] = it - out.begin()

        x = cupy.array([0, 1, 2, 4, 6, 7])
        y = cupy.array([1, 2, 5, 8])
        out = cupy.zeros((10,), dtype=numpy.int64)
        size = cupy.zeros((1,), dtype=numpy.int64)
        set_union[1, 1](x, y, out, size)

        testing.assert_array_equal(size, cupy.array([8]))
        testing.assert_array_equal(
            out[:8], cupy.array([0, 1, 2, 4, 5, 6, 7, 8]))

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_set_union_by_key(self, order):
        if runtime.is_hip:
            pytest.xfail('HIP does not support thrust set operations.')

        @jit.rawkernel()
        def set_union_by_key(
                keys1, keys2, values1, values2, keys_out, values_out, size):
            it0, it1 = jit.thrust.set_union_by_key(
                jit.thrust.device,
                keys1.begin(), keys1.end(),
                keys2.begin(), keys2.end(),
                values1.begin(), values2.begin(),
                keys_out.begin(), values_out.begin()
            )
            size[0] = it0 - keys_out.begin()
            size[1] = it1 - values_out.begin()

        a_keys = cupy.array([0, 1, 2, 4, 6, 7])
        a_vals = cupy.array([1, 1, 1, 1, 1, 1])
        b_keys = cupy.array([1, 2, 5, 8])
        b_vals = cupy.array([2, 2, 2, 2])
        keys_out = cupy.zeros((10,), dtype=numpy.int64)
        vals_out = cupy.zeros((10,), dtype=numpy.int64)
        size = cupy.zeros((2,), dtype=numpy.int64)
        set_union_by_key[1, 1](
            a_keys, b_keys, a_vals, b_vals, keys_out, vals_out, size)

        testing.assert_array_equal(size, cupy.array([8, 8]))
        testing.assert_array_equal(
            keys_out[:8], cupy.array([0, 1, 2, 4, 5, 6, 7, 8]))
        testing.assert_array_equal(
            vals_out[:8], cupy.array([1, 1, 1, 1, 2, 1, 1, 2]))

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_sort(self, order):
        if runtime.is_hip:
            pytest.skip('See https://github.com/cupy/cupy/pull/7162')

        @jit.rawkernel()
        def sort(x):
            i = jit.threadIdx.x
            array = x[i]
            jit.thrust.sort(jit.thrust.device, array.begin(), array.end())

        h, w = (256, 256)
        x = testing.shaped_random(
            (h, w), dtype=numpy.int32, scale=4, order=order)
        x_numpy = cupy.asnumpy(x)
        sort[1, 256](x)

        testing.assert_array_equal(x, numpy.sort(x_numpy, axis=-1))

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_sort_by_key(self, order):
        if runtime.is_hip:
            pytest.skip('See https://github.com/cupy/cupy/pull/7162')

        @jit.rawkernel()
        def sort_by_key(x, y):
            i = jit.threadIdx.x
            x_array = x[i]
            y_array = y[i]
            jit.thrust.sort_by_key(
                jit.thrust.device,
                x_array.begin(),
                x_array.end(),
                y_array.begin(),
            )

        h, w = (256, 256)
        x = cupy.arange(h * w, dtype=numpy.int32)
        cupy.random.shuffle(x)
        x = x.reshape(h, w)
        y = testing.shaped_random(
            (h, w), dtype=numpy.int32, scale=20000, order=order, seed=1)
        x_numpy = cupy.asnumpy(x)
        y_numpy = cupy.asnumpy(y)
        sort_by_key[1, 256](x, y)

        indices = numpy.argsort(x_numpy, axis=-1)
        x_expected = numpy.array([a[i] for a, i in zip(x_numpy, indices)])
        y_expected = numpy.array([a[i] for a, i in zip(y_numpy, indices)])
        testing.assert_array_equal(x, x_expected)
        testing.assert_array_equal(y, y_expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_stable_sort(self, order):
        if runtime.is_hip:
            pytest.skip('See https://github.com/cupy/cupy/pull/7162')

        @jit.rawkernel()
        def stable_sort(x):
            i = jit.threadIdx.x
            array = x[i]
            jit.thrust.stable_sort(
                jit.thrust.device, array.begin(), array.end())

        h, w = (256, 256)
        x = testing.shaped_random(
            (h, w), dtype=numpy.int32, scale=4, order=order)
        x_numpy = cupy.asnumpy(x)
        stable_sort[1, 256](x)

        testing.assert_array_equal(x, numpy.sort(x_numpy, axis=-1))

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_stable_sort_by_key(self, order):
        if runtime.is_hip:
            pytest.skip('See https://github.com/cupy/cupy/pull/7162')

        @jit.rawkernel()
        def stable_sort_by_key(x, y):
            i = jit.threadIdx.x
            x_array = x[i]
            y_array = y[i]
            jit.thrust.stable_sort_by_key(
                jit.thrust.device,
                x_array.begin(),
                x_array.end(),
                y_array.begin(),
            )

        h, w = (256, 256)
        x = cupy.arange(h * w, dtype=numpy.int32)
        cupy.random.shuffle(x)
        x = x.reshape(h, w)
        y = testing.shaped_random(
            (h, w), dtype=numpy.int32, scale=20000, order=order, seed=1)
        x_numpy = cupy.asnumpy(x)
        y_numpy = cupy.asnumpy(y)
        stable_sort_by_key[1, 256](x, y)

        indices = numpy.argsort(x_numpy, axis=-1)
        x_expected = numpy.array([a[i] for a, i in zip(x_numpy, indices)])
        y_expected = numpy.array([a[i] for a, i in zip(y_numpy, indices)])
        testing.assert_array_equal(x, x_expected)
        testing.assert_array_equal(y, y_expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_swap_ranges(self, order):
        if runtime.is_hip:
            pytest.skip('See https://github.com/cupy/cupy/pull/7162')

        @jit.rawkernel()
        def swap_ranges(x, y):
            i = jit.threadIdx.x
            jit.thrust.swap_ranges(
                jit.thrust.device, x[i].begin(), x[i].end(), y[i].begin())

        h, w = (256, 256)
        x = testing.shaped_random(
            (h, w), dtype=numpy.int32, scale=4, order=order, seed=0)
        y = testing.shaped_random(
            (h, w), dtype=numpy.int32, scale=4, order=order, seed=1)
        x_expected = y.copy()
        y_expected = x.copy()
        swap_ranges[1, 256](x, y)

        testing.assert_array_equal(x_expected, x)
        testing.assert_array_equal(y_expected, y)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_unique(self, order):

        @jit.rawkernel()
        def unique(x, size):
            it = jit.thrust.unique(jit.thrust.device, x.begin(), x.end())
            size[0] = it - x.begin()

        x = cupy.array([1, 3, 3, 3, 2, 2, 1])
        size = cupy.zeros((1,), dtype=numpy.int32)
        unique[1, 1](x, size)

        testing.assert_array_equal(size, cupy.array([4], dtype=numpy.int32))
        testing.assert_array_equal(x[:4], cupy.array([1, 3, 2, 1]))

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_unique_by_key(self, order):

        @jit.rawkernel()
        def unique_by_key(keys, values, size):
            it0, it1 = jit.thrust.unique_by_key(
                jit.thrust.device, keys.begin(), keys.end(), values.begin())
            size[0] = it0 - keys.begin()
            size[1] = it1 - values.begin()

        keys = cupy.array([1, 3, 3, 3, 2, 2, 1])
        values = cupy.array([9, 8, 7, 6, 5, 4, 3])
        size = cupy.zeros((2,), dtype=numpy.int32)
        unique_by_key[1, 1](keys, values, size)

        testing.assert_array_equal(size, cupy.array([4, 4], dtype=numpy.int32))
        testing.assert_array_equal(keys[:4], cupy.array([1, 3, 2, 1]))
        testing.assert_array_equal(values[:4], cupy.array([9, 8, 5, 3]))

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_upper_bound(self, order):
        @jit.rawkernel()
        def upper_bound(x, v, out):
            i = jit.threadIdx.x
            it = jit.thrust.upper_bound(
                jit.thrust.seq, x.begin(), x.end(), v[i])
            out[i] = it - x.begin()

        n1, n2 = (128, 160)
        x = testing.shaped_random((n2,), dtype=numpy.int32, order=order)
        x = cupy.sort(x)
        values = testing.shaped_random((n1,), dtype=numpy.int32, order=order)
        out = cupy.zeros(n1, dtype=numpy.int32)
        upper_bound[1, n1](x, values, out)

        expected = cupy.searchsorted(x, values, side='right')
        testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_upper_bound_vec(self, order):
        @jit.rawkernel()
        def upper_bound(x, v, out):
            i = jit.threadIdx.x
            jit.thrust.upper_bound(
                jit.thrust.seq, x.begin(), x.end(),
                v[i].begin(), v[i].end(), out[i].begin())

        n1, n2, n3 = (128, 160, 200)
        x = testing.shaped_random(
            (n2,), dtype=numpy.int32, scale=200, order=order, seed=0)
        x = cupy.sort(x)
        values = testing.shaped_random(
            (n1, n3), dtype=numpy.int32, scale=200, order=order, seed=1)
        out = cupy.zeros((n1, n3), dtype=numpy.int32)
        upper_bound[1, n1](x, values, out)

        expected = cupy.searchsorted(x, values, side='right')
        testing.assert_array_equal(out, expected)
