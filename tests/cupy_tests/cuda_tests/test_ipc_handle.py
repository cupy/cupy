import unittest
import cupy
from cupy import testing
from multiprocessing import get_context
import pytest


class TestIPCHandle(unittest.TestCase):

    def setUp(self):
        self.context = get_context("spawn")

    @staticmethod
    def modify_array(handle, result_queue):
        arr = handle.open()
        arr *= 2
        result_queue.put(arr)

    @staticmethod
    def get_twice(handle, error_queue):
        try:
            handle.open()
            handle.open()

        except RuntimeError as e:
            error_queue.put(e)

    def test_IPCHandle_get(self):
        handle = cupy.ones(10).get_ipc_handle()

        result_queue = self.context.Queue()
        p = self.context.Process(target=self.modify_array,
                                 args=(handle, result_queue))
        p.start()
        p.join()

        result = cupy.ones(10)*2
        testing.assert_array_equal(result, result_queue.get())

    def test_IPCHandle_error_handling(self):
        arr = cupy.ones(10)
        handle = arr.get_ipc_handle()

        with pytest.raises(RuntimeError):
            handle.open()

        error_queue = self.context.Queue()
        p = self.context.Process(target=self.get_twice,
                                 args=(handle, error_queue))
        p.start()
        p.join()
        assert not error_queue.empty(), "No RuntimeError was raised"
        assert isinstance(error_queue.get(), RuntimeError)
