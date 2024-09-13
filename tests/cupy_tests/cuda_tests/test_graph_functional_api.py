import sys
import os
import unittest

import pytest
import cupy
from cupy import cuda
from cupy.cuda.graph_functional_api import (
    GraphBuilderInterface,
    GraphBuilder,
    MockGraphBuilder
)

@pytest.mark.skipif(cuda.runtime.is_hip,
                    reason='HIP does not support this')
@pytest.mark.skipif(cuda.get_local_runtime_version() < 12040,
                    reason='Conditional API requires CUDA >=12.4')
class TestGraphFunctionalAPI(unittest.TestCase):
    def setUp(self):
        # Setup environment variable to allow graph capture
        self.prev_cublas_allow_capture = \
            cuda.cublas._allow_stream_graph_capture
        self.prev_cusparse_allow_capture = \
            cuda.cusparse._allow_stream_graph_capture
        cuda.cublas._allow_stream_graph_capture = True
        cuda.cusparse._allow_stream_graph_capture = True

    def tearDown(self):
        cuda.cublas._allow_stream_graph_capture = \
            self.prev_cublas_allow_capture
        cuda.cusparse._allow_stream_graph_capture = \
            self.prev_cusparse_allow_capture

    def test_simple_while(self):
        def tester(gb: GraphBuilderInterface):
            @gb.graphify
            def test_func(a: cupy.ndarray, N: int):
                def while_fn(a):
                    a += 1
                    return (a,)
                a, = gb.while_loop(
                    cond_fn=lambda a: cupy.all(a != N),
                    body_fn=while_fn,
                    fn_args=(a,)
                )
                return a

            size = 100000
            N = 1000
            a = cupy.zeros(size, dtype=cupy.int32)
            return test_func(a, N)

        result_graph = tester(GraphBuilder())
        result_nograph = tester(MockGraphBuilder())

        assert cupy.all(result_graph == result_nograph)

    def test_simple_if_true(self):
        def tester(gb: GraphBuilderInterface, a_in, b_in):
            a = cupy.copy(a_in)
            b = cupy.copy(b_in)
            # if one.dtype is cupy.float32, this test fails
            one = cupy.ones((), dtype=cupy.bool_)

            @gb.graphify
            def test_func(a, b, condition):
                def if_true():
                    a[...] += b
                gb.cond(
                    lambda: condition,
                    if_true
                )
                return a

            return test_func(a, b, one)

        a = cupy.zeros(100)
        b = cupy.arange(100)
        result_mock = tester(MockGraphBuilder(), a, b)
        result_graph = tester(GraphBuilder(), a, b)

        assert cupy.all(result_graph == result_mock)

    def test_simple_if_false(self):
        def tester(
            gb: GraphBuilderInterface, a_in, b_in
        ):
            a = cupy.copy(a_in)
            b = cupy.copy(b_in)
            zero = cupy.zeros((), dtype=cupy.bool_)

            @gb.graphify
            def test_func(a, b, condition):
                def if_true():
                    a[...] += b
                gb.cond(
                    lambda: condition,
                    if_true
                )
                return a

            return test_func(a, b, zero)

        a = cupy.zeros(100)
        b = cupy.arange(100)
        result_mock = tester(MockGraphBuilder(), a, b)
        result_graph = tester(GraphBuilder(), a, b)

        assert cupy.all(result_graph == result_mock)

    def test_nested_while(self):
        size = 500
        loop = 10
        def tester(
            gb: GraphBuilderInterface, A_in, x_in
        ):
            A = cupy.copy(A_in)
            x = cupy.copy(x_in)

            counter = cupy.zeros((), dtype=cupy.int32)
            @gb.graphify
            def test_func(A, x, counter):
                def while_out(A, x, counter):
                    counter += 1
                    def while_in(A, x):
                        cupy.matmul(A, x, out=x)
                        return (A, x)
                    gb.while_loop(
                        cond_fn=lambda *_: (cupy.linalg.norm(x) / size) < size,
                        body_fn=while_in,
                        fn_args=(A, x)
                    )
                    x /= size
                    return (A, x, counter)
                gb.while_loop(
                    lambda *_: counter < loop,
                    body_fn=while_out,
                    fn_args=(A, x, counter)
                )

                return x

            return test_func(A, x, counter)

        def normal_cupy_impl(A_in, x_in):
            A = cupy.copy(A_in)
            x = cupy.copy(x_in)

            counter = cupy.zeros((), dtype=cupy.int32)
            while counter < loop:
                counter += 1
                while cupy.linalg.norm(x) / size < size:
                    x = A @ x
                x /= size
            return x

        cupy.random.seed(42)
        A = cupy.random.randn(size, size)
        x = cupy.random.randn(size, 1)

        result_mock = tester(MockGraphBuilder(), A, x)
        result_graph = tester(GraphBuilder(), A, x)
        result = normal_cupy_impl(A, x)
        assert cupy.all(result_graph == result_mock)
        assert cupy.all(result_graph == result)

    def test_kmeans(self):
        RANDOM_SEED = 42
        num = 50000
        n_clusters = 2
        max_iter = 1000

        def kmeans_tester(
            gb: GraphBuilderInterface, X_in, initials_in
        ):
            # Array setup
            X = cupy.copy(X_in)

            n_samples = len(X)
            pred = cupy.zeros(n_samples)
            i = cupy.arange(n_clusters)
            initial_indexes = cupy.copy(initials_in)
            centers = X[initial_indexes]
            cond = cupy.ones((), dtype=cupy.bool_)

            # Initial prediction
            distances = cupy.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            new_pred = cupy.argmin(distances, axis=1)

            @gb.graphify
            def kmeans(pred, new_pred, centers, distances):
                def while_fn(pred, new_pred, centers, distances):
                    cupy.copyto(pred, new_pred)

                    mask = pred == i[:, None]
                    sums = cupy.where(mask[:, :, None], X, 0).sum(axis=1)
                    counts = cupy.count_nonzero(mask, axis=1).reshape((n_clusters, 1))
                    centers = sums / counts

                    distances = cupy.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
                    cupy.argmin(distances, axis=1, out=new_pred)
                    return (pred, new_pred, centers, distances)
                pred, _, centers, _ = gb.while_loop(
                    cond_fn=lambda *_: cupy.any(new_pred != pred),
                    body_fn=while_fn,
                    fn_args=(pred, new_pred, centers, distances)
                )

                return centers, pred

            return kmeans(pred, new_pred, centers, distances)

        def fit_normal_cupy(X_in, initials_in):
            X = cupy.copy(X_in)
            n_samples = len(X)

            pred = cupy.zeros(n_samples)
            initial_indexes = cupy.copy(initials_in)
            centers = X[initial_indexes]

            i = cupy.arange(n_clusters)

            distances = cupy.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            new_pred = cupy.argmin(distances, axis=1)

            while cupy.any(new_pred != pred):
                cupy.copyto(pred, new_pred)

                mask = pred == i[:, None]
                sums = cupy.where(mask[:, :, None], X, 0).sum(axis=1)
                counts = cupy.count_nonzero(mask, axis=1).reshape((n_clusters, 1))
                centers = sums / counts

                distances = cupy.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
                new_pred = cupy.argmin(distances, axis=1)

            return centers, pred

        cupy.random.seed(RANDOM_SEED)
        samples = cupy.random.randn(num, 2)
        X_in = cupy.concatenate((samples + 1, samples - 1))
        n_samples = len(X_in)
        initial_indexes = cupy.random.choice(n_samples, n_clusters, replace=False)

        # k-means in normal CuPy
        centers_normal, pred_normal = fit_normal_cupy(X_in, initial_indexes)
        centers_mock, pred_mock = kmeans_tester(MockGraphBuilder(), X_in, initial_indexes)
        centers_graph, pred_graph = kmeans_tester(GraphBuilder(), X_in, initial_indexes)

        # graph == normal
        assert cupy.allclose(centers_graph, centers_normal)
        assert cupy.all(pred_graph == pred_normal)

        # graph == mock
        assert cupy.allclose(centers_graph, centers_mock)
        assert cupy.all(pred_graph == pred_mock)

    def test_multicond(self):
        x = cupy.array([1, 1, 1])
        y = cupy.array([2, 2, 2])
        z = cupy.array([3, 3, 3])
        w = cupy.array([4, 4, 4])
        def impl(gb: GraphBuilderInterface, array):
            array = array.copy()

            @gb.graphify
            def target(array):
                output = x.copy()
                def fn0():
                    cupy.copyto(output, x)
                def fn1():
                    cupy.copyto(output, y)
                def fn2():
                    cupy.copyto(output, z)
                def fn3():
                    cupy.copyto(output, w)
                gb.multicond([
                    (lambda: array[0], fn0),
                    (lambda: array[1], fn1),
                    (lambda: array[2], fn2),
                    (None, fn3),
                ])
                return output

            return target(array)

        def tester(arr, ans):
            out_graph = impl(GraphBuilder(), arr)
            out_mock = impl(MockGraphBuilder(), arr)
            assert cupy.all(out_graph == ans)
            assert cupy.all(out_mock == ans)

        tester(cupy.array([True, False, False]), x)
        tester(cupy.array([True, True, False]), x)
        tester(cupy.array([True, True, True]), x)
        tester(cupy.array([True, False, True]), x)

        tester(cupy.array([False, True, True]), y)
        tester(cupy.array([False, True, False]), y)

        tester(cupy.array([False, False, True]), z)
        tester(cupy.array([False, False, False]), w)

    def test_cublas_capture(self):
        # If cuBLAS's workspace is not set properly, this test will fail
        # See: https://docs.nvidia.com/cuda/cublas/#cuda-graphs-support
        from cupy import cublas
        out = cupy.empty((), dtype=cupy.float64)
        x = cupy.ones(100, dtype=cupy.float64)

        gb = GraphBuilder()
        @gb.graphify
        def use_cublas():
            def inner():
                # This function uses stream-ordered allocation unless
                # cuBLAS workspace is not set, which is incompatible with
                # child graph
                cublas.dot(x, x, out)
            gb.cond(
                lambda: cupy.ones((), dtype=cupy.bool_),
                inner,
            )

        use_cublas()
