import unittest

import pytest

import cupy
from cupy import testing


class TestDLPackConversion(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    def test_conversion(self, dtype):
        self.dtype = dtype
        if cupy.issubdtype(self.dtype, cupy.unsignedinteger):
            self.array = cupy.random.randint(
                0, 10, size=(2, 3)).astype(self.dtype)
        elif cupy.issubdtype(self.dtype, cupy.integer):
            self.array = cupy.random.randint(
                -10, 10, size=(2, 3)).astype(self.dtype)
        elif cupy.issubdtype(self.dtype, cupy.floating):
            self.array = cupy.random.rand(
                2, 3).astype(self.dtype)

        tensor = self.array.toDlpack()
        array = cupy.fromDlpack(tensor)
        testing.assert_array_equal(self.array, array)
        testing.assert_array_equal(self.array.data.ptr, array.data.ptr)


class TestDLTensorMemory(unittest.TestCase):

    def setUp(self):
        self.old_pool = cupy.get_default_memory_pool()
        self.pool = cupy.cuda.MemoryPool()
        cupy.cuda.set_allocator(self.pool.malloc)

    def tearDown(self):
        self.pool.free_all_blocks()
        cupy.cuda.set_allocator(self.old_pool.malloc)

    def test_deleter(self):
        # memory is freed when tensor is deleted, as it's not consumed
        array = cupy.empty(10)
        tensor = array.toDlpack()
        # str(tensor): <capsule object "dltensor" at 0x7f7c4c835330>
        assert "\"dltensor\"" in str(tensor)
        assert self.pool.n_free_blocks() == 0
        del array
        assert self.pool.n_free_blocks() == 0
        del tensor
        assert self.pool.n_free_blocks() == 1

    def test_deleter2(self):
        # memory is freed when array2 is deleted, as tensor is consumed
        array = cupy.empty(10)
        tensor = array.toDlpack()
        assert "\"dltensor\"" in str(tensor)
        array2 = cupy.fromDlpack(tensor)
        assert "\"used_dltensor\"" in str(tensor)
        assert self.pool.n_free_blocks() == 0
        del array
        assert self.pool.n_free_blocks() == 0
        del array2
        assert self.pool.n_free_blocks() == 1
        del tensor
        assert self.pool.n_free_blocks() == 1

    def test_multiple_consumption_error(self):
        # Prevent segfault, see #3611
        array = cupy.empty(10)
        tensor = array.toDlpack()
        array2 = cupy.fromDlpack(tensor)  # noqa
        with pytest.raises(ValueError) as e:
            array3 = cupy.fromDlpack(tensor)  # noqa
        assert 'consumed multiple times' in str(e.value)
