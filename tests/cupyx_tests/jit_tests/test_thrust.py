import numpy

import pytest

import cupy
from cupy import testing
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
    def test_find_iterator(self, order):
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
    def test_sort_iterator(self, order):
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
    def test_sort_by_key_iterator(self, order):
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
