import contextlib
import io
import queue
import threading
import unittest

import numpy as np
import pytest

import cupy
from cupy import testing
from cupy.cuda import cufft
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.fft import config


class TestPlanCache(unittest.TestCase):
    def setUp(self):
        config.clear_plan_cache()
        self.old_size = config.get_plan_cache_size()
        config.set_plan_cache_size(2)

    def tearDown(self):
        config.clear_plan_cache()
        config.set_plan_cache_size(self.old_size)

    def test_LRU_cache1(self):
        # test if insertion and clean-up works
        cache = config.get_plan_cache()
        assert cache.get_curr_size() == 0 <= cache.get_size()
        a = testing.shaped_random((10,), cupy, cupy.float32)
        b = cupy.fft.fft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()
        cache.clear()
        assert cache.get_curr_size() == 0 <= cache.get_size()

    def test_LRU_cache2(self):
        # test if plan is reused
        cache = config.get_plan_cache()
        assert cache.get_curr_size() == 0 <= cache.get_size()

        # run once and fetch the cached plan
        a = testing.shaped_random((10,), cupy, cupy.float32)
        b = cupy.fft.fft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()
        iterator = iter(cache)
        plan0 = next(iterator)[1].plan

        # repeat
        b = cupy.fft.fft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()
        iterator = iter(cache)
        plan1 = next(iterator)[1].plan

        # we should get the same plan
        assert plan0 is plan1

    def test_LRU_cache3(self):
        # test if cache size is limited
        cache = config.get_plan_cache()
        assert cache.get_curr_size() == 0 <= cache.get_size()

        # run once and fetch the cached plan
        a = testing.shaped_random((10,), cupy, cupy.float32)
        b = cupy.fft.fft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()
        iterator = iter(cache)
        plan = next(iterator)[1].plan

        # run another two FFTs with different sizes so that the first
        # plan is discarded from the cache
        a = testing.shaped_random((20,), cupy, cupy.float32)
        b = cupy.fft.fft(a)
        assert cache.get_curr_size() == 2 <= cache.get_size()
        a = testing.shaped_random((30,), cupy, cupy.float32)
        b = cupy.fft.fft(a)
        assert cache.get_curr_size() == 2 <= cache.get_size()

        # check if the first plan is indeed not cached
        for _, node in cache:
            assert plan is not node.plan

    def test_LRU_cache4(self):
        # test if fetching the plan will reorder it to the top
        cache = config.get_plan_cache()
        assert cache.get_curr_size() == 0 <= cache.get_size()

        # this creates a Plan1d
        a = testing.shaped_random((10,), cupy, cupy.float32)
        b = cupy.fft.fft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()

        # this creates a PlanNd
        a = testing.shaped_random((10, 20), cupy, cupy.float32)
        b = cupy.fft.fftn(a)
        assert cache.get_curr_size() == 2 <= cache.get_size()

        # The first in the cache is the most recently used one;
        # using an iterator to access the linked list guarantees that
        # we don't alter the cache order
        iterator = iter(cache)
        assert isinstance(next(iterator)[1].plan, cufft.PlanNd)
        assert isinstance(next(iterator)[1].plan, cufft.Plan1d)
        with pytest.raises(StopIteration):
            next(iterator)

        # this brings Plan1d to the top
        a = testing.shaped_random((10,), cupy, cupy.float32)
        b = cupy.fft.fft(a)
        assert cache.get_curr_size() == 2 <= cache.get_size()
        iterator = iter(cache)
        assert isinstance(next(iterator)[1].plan, cufft.Plan1d)
        assert isinstance(next(iterator)[1].plan, cufft.PlanNd)
        with pytest.raises(StopIteration):
            next(iterator)

        # An LRU cache guarantees that such a silly operation never
        # raises StopIteration
        iterator = iter(cache)
        for i in range(100):
            cache[next(iterator)[0]]

    @testing.multi_gpu(2)
    def test_LRU_cache5(self):
        # test if the LRU cache is thread-local

        def init_caches(gpus):
            for i in gpus:
                with device.Device(i):
                    cache = config.get_plan_cache()

        def intercept_stdout():
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                config.show_plan_cache_info()
                stdout = buf.getvalue()
            return stdout

        # Testing in the current thread is tricky as we don't know if
        # the cache for device 2 is initialized or not by this point.
        # Let us force its initialization.
        n_devices = runtime.getDeviceCount()
        init_caches(range(n_devices))
        stdout = intercept_stdout()
        assert 'uninitialized' not in stdout

        def thread_show_plan_cache_info(queue):
            # allow output from another thread to be accessed by the
            # main thread
            stdout = intercept_stdout()
            queue.put(stdout)

        # When starting a new thread, the cache is uninitialized there
        # (for both devices)
        q = queue.Queue()
        thread = threading.Thread(target=thread_show_plan_cache_info,
                                   args=(q,))
        thread.start()
        thread.join()
        stdout = q.get()
        assert stdout.count('uninitialized') == 2

        def thread_init_caches(gpus, queue):
            init_caches(gpus)
            thread_show_plan_cache_info(queue)

        # Now let's try initializing device 0 on another thread
        thread = threading.Thread(target=thread_init_caches,
                                   args=([0], q,))
        thread.start()
        thread.join()
        stdout = q.get()
        assert stdout.count('uninitialized') == 1

        # ...and this time both devices
        thread = threading.Thread(target=thread_init_caches,
                                   args=([0, 1], q,))
        thread.start()
        thread.join()
        stdout = q.get()
        assert stdout.count('uninitialized') == 0

    @testing.multi_gpu(2)
    def test_LRU_cache6(self):
        # test if each device has a seperate cache
        with device.Device(0):
            cache0 = config.get_plan_cache()
            cache0.clear()
            cache0.set_size(2)
        with device.Device(1):
            cache1 = config.get_plan_cache()
            cache1.clear()
            cache1.set_size(2)

        # ensure a fresh state
        assert cache0.get_curr_size() == 0 <= cache0.get_size()
        assert cache1.get_curr_size() == 0 <= cache1.get_size()

        # do some computation on GPU 0
        with device.Device(0):
            a = testing.shaped_random((10,), cupy, cupy.float32)
            b = cupy.fft.fft(a)
        assert cache0.get_curr_size() == 1 <= cache0.get_size()
        assert cache1.get_curr_size() == 0 <= cache1.get_size()

        # do some computation on GPU 1
        with device.Device(1):
            c = testing.shaped_random((16,), cupy, cupy.float64)
            d = cupy.fft.fft(c)
        assert cache0.get_curr_size() == 1 <= cache0.get_size()
        assert cache1.get_curr_size() == 1 <= cache1.get_size()

        # reset device 0
        cache0.clear()
        assert cache0.get_curr_size() == 0 <= cache0.get_size()
        assert cache1.get_curr_size() == 1 <= cache1.get_size()

        # reset device 1
        cache1.clear()
        assert cache0.get_curr_size() == 0 <= cache0.get_size()
        assert cache1.get_curr_size() == 0 <= cache1.get_size()

    @testing.multi_gpu(2)
    def test_LRU_cache7(self):
        # test accessing a multi-GPU plan
        pass

    def test_LRU_cache8(self):
        # test if Plan1d and PlanNd can coexist in the same cache
        pass

    def test_LRU_cache9(self):
        # test if memsizes in the cache adds up
        pass
