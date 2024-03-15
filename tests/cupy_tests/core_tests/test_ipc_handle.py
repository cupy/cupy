import unittest
import cupy
from cupy import testing
from multiprocessing import get_context
import pytest


class TestIPCHandle(unittest.TestCase):

    def setUp(self):
        self.spawn_context = get_context("spawn")
        self.fork_context = get_context("fork")

    @staticmethod
    def modify_array(handle, error_queue):
        try:
            arr = handle.get()
            arr *= 2

        except RuntimeError as e:
            error_queue.put(e)

    @staticmethod
    def get_twice(handle, error_queue):
        try:
            handle.get()
            handle.get()

        except RuntimeError as e:
            error_queue.put(e)

    def test_IPCHandle_get(self):
        arr = cupy.ones(10)
        handle = arr.get_ipc_handle()

        error_queue = self.spawn_context.Queue()
        p = self.spawn_context.Process(target=self.modify_array,
                                 args=(handle, error_queue))
        p.start()
        p.join()

        result = cupy.ones(10)*2
        testing.assert_array_equal(result, arr)

    def test_IPCHandle_error_handling(self):
        arr = cupy.ones(10)
        handle = arr.get_ipc_handle()

        with pytest.raises(RuntimeError):
            handle.get()

        error_queue = self.fork_context.Queue()
        p = self.fork_context.Process(target=self.modify_array,
                                 args=(handle, error_queue))
        p.start()
        p.join()
        assert not error_queue.empty(), "No RuntimeError was raised"
        assert isinstance(error_queue.get(), RuntimeError)

        error_queue = self.spawn_context.Queue()
        p = self.spawn_context.Process(target=self.get_twice,
                                 args=(handle, error_queue))
        p.start()
        p.join()
        assert not error_queue.empty(), "No RuntimeError was raised"
        assert isinstance(error_queue.get(), RuntimeError)
