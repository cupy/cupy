import pickle
import threading
import unittest

import cupy
from cupy.cuda import driver


@unittest.skipIf(cupy.cuda.runtime.is_hip, 'Context API is dperecated in HIP')
class TestDriver(unittest.TestCase):
    def test_ctxGetCurrent(self):
        # Make sure to create context.
        cupy.arange(1)
        assert 0 != driver.ctxGetCurrent()

    def test_ctxGetCurrent_thread(self):
        # Make sure to create context in main thread.
        cupy.arange(1)

        def f(self):
            self._result0 = driver.ctxGetCurrent()
            cupy.arange(1)
            self._result1 = driver.ctxGetCurrent()

        self._result0 = None
        self._result1 = None
        t = threading.Thread(target=f, args=(self,))
        t.daemon = True
        t.start()
        t.join()

        # The returned context pointer must be NULL on sub thread
        # without valid context.
        assert 0 == self._result0

        # After the context is created, it should return the valid
        # context pointer.
        assert 0 != self._result1


class TestExceptionPicklable(unittest.TestCase):

    def test(self):
        e1 = driver.CUDADriverError(1)
        e2 = pickle.loads(pickle.dumps(e1))
        assert e1.args == e2.args
        assert str(e1) == str(e2)
