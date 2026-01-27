from __future__ import annotations

import concurrent.futures
import gc
import random
import threading

import pytest

import cupy
from cupy.cuda.memory import alloc


pytestmark = pytest.mark.thread_unsafe(
    reason="tests in this module are already explicitly multi-threaded"
)


def run_threaded(func, max_workers=8, pass_count=False,
                 pass_barrier=False, outer_iterations=1,
                 prepare_args=None):
    """Runs a function many times in parallel

    This function has been taken from NumPy:
    https://github.com/numpy/numpy/blob/a90ef57574c501a780fe834123b20fcea1329f90/numpy/testing/_private/utils.py#L2807
    """
    for _ in range(outer_iterations):
        with (concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
              as tpe):
            if prepare_args is None:
                args = []
            else:
                args = prepare_args()
            if pass_barrier:
                barrier = threading.Barrier(max_workers)
                args.append(barrier)
            if pass_count:
                all_args = [(func, i, *args) for i in range(max_workers)]
            else:
                all_args = [(func, *args) for i in range(max_workers)]
            try:
                futures = []
                for arg in all_args:
                    futures.append(tpe.submit(*arg))
            except RuntimeError as e:
                import pytest
                pytest.skip(f"Spawning {max_workers} threads failed with "
                            f"error {e!r} (likely due to resource limits on "
                            "the system running the tests)")
            finally:
                if len(futures) < max_workers and pass_barrier:
                    barrier.abort()
            for f in futures:
                f.result()


@pytest.mark.slow
def test_elementwise_kernel_cache():
    """Checks that a thread always uses the same compiled kernel
    which means that we don't unload a kernel that was ever used.

    This matters for graph capture, although in some cases just
    unloading a module may create problems and that isn't prevented.
    I.e. a race can still mean that multiple threads compile the same code.
    When this happens, we test that one version is used everywhere.
    """
    def prepare_args():
        kernel = cupy.ElementwiseKernel("T x", "T y", "y = x;")
        assert not kernel._elementwise_kernel_memo
        arr = cupy.ones(10)
        return [kernel, arr]

    def func(kernel, arr, barrier):
        barrier.wait()
        kernel(arr)
        assert len(kernel._elementwise_kernel_memo) == 1
        cached_obj1 = next(iter(kernel._elementwise_kernel_memo.values()))
        kernel(arr)
        assert len(kernel._elementwise_kernel_memo) == 1
        cached_obj2 = next(iter(kernel._elementwise_kernel_memo.values()))
        assert cached_obj1 is cached_obj2

    run_threaded(func, outer_iterations=20,
                 pass_barrier=True, prepare_args=prepare_args)


@pytest.mark.slow
def test_ufunc_kernel_cache():
    # See test_elementwise_kernel_cache for more details.
    def prepare_args():
        ufunc = cupy._core.create_ufunc("cache_test", ("d->d",), "out0 = in0")
        assert not ufunc._kernel_memo
        arr = cupy.ones(10)
        return [ufunc, arr]

    def func(ufunc, arr, barrier):
        barrier.wait()
        ufunc(arr)
        assert len(ufunc._kernel_memo) == 1
        cached_obj1 = next(iter(ufunc._kernel_memo.values()))
        ufunc(arr)
        assert len(ufunc._kernel_memo) == 1
        cached_obj2 = next(iter(ufunc._kernel_memo.values()))
        assert cached_obj1 is cached_obj2

    run_threaded(func, outer_iterations=20,
                 pass_barrier=True, prepare_args=prepare_args)


@pytest.mark.slow
# NOTE: With clean=False, this test can OOM, since the cycles may not
# be cleaned up sufficiently in the `gc.collect()` we do on OOM.
@pytest.mark.parametrize("clean", [True, False])
def test_default_memory_pool_threaded(clean, iterations=500):
    # This test is designed to stress-test the memory pool, we will
    # create various usage patterns and mix them in a threaded way.
    # To seriously stress-test it make the iterations very large and watch
    # the long-term behavior.

    def random_allocation():
        # choose a random allocation size, hopefully this will (occasionally)
        # lead to allocations being split.
        size = random.randint(1, 50_000)
        return alloc(size)

    def make_allocations():
        allocations = []
        for i in range(random.randint(1, 50)):
            allocations.append(random_allocation())

        # And now let's make a few that can't be cleand up easily.
        first = [None, random_allocation()]
        curr = first
        for i in range(2, 50):
            node = [curr, random_allocation()]
            curr = node

        first[0] = curr  # close the circle

        return allocations

    def func():
        for i in range(iterations):  # increase to test for longer
            _ = make_allocations()
            # once in a while, we either collect or free all blocks
            # to stress those paths more. But hitting the high-water mark
            # with clean=False is also interesting.
            if clean:
                if i % 10 == 0:
                    gc.collect()
                elif i % 10 == 5:
                    cupy.get_default_memory_pool().free_all_blocks()
            _ = make_allocations()

    run_threaded(func)
