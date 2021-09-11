import unittest

import pytest

import cupy
from cupy import cuda
from cupy import testing


def _gen_array(dtype):
    if cupy.issubdtype(dtype, cupy.unsignedinteger):
        array = cupy.random.randint(
            0, 10, size=(2, 3)).astype(dtype)
    elif cupy.issubdtype(dtype, cupy.integer):
        array = cupy.random.randint(
            -10, 10, size=(2, 3)).astype(dtype)
    elif cupy.issubdtype(dtype, cupy.floating):
        array = cupy.random.rand(
            2, 3).astype(dtype)
    elif cupy.issubdtype(dtype, cupy.complexfloating):
        array = cupy.random.random((2, 3)).astype(dtype)
    else:
        assert False, f'unrecognized dtype: {dtype}'
    return array


class TestDLPackConversion(unittest.TestCase):

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    @testing.for_all_dtypes(no_bool=True)
    def test_conversion(self, dtype):
        orig_array = _gen_array(dtype)
        tensor = orig_array.toDlpack()
        out_array = cupy.fromDlpack(tensor)
        testing.assert_array_equal(orig_array, out_array)
        testing.assert_array_equal(orig_array.data.ptr, out_array.data.ptr)


class TestNewDLPackConversion(unittest.TestCase):

    def _get_stream(self, stream_name):
        if stream_name == 'null':
            return cuda.Stream.null
        elif stream_name == 'ptds':
            return cuda.Stream.ptds
        else:
            return cuda.Stream()

    @testing.for_all_dtypes(no_bool=True)
    def test_conversion(self, dtype):
        orig_array = _gen_array(dtype)
        out_array = cupy.from_dlpack(orig_array)
        testing.assert_array_equal(orig_array, out_array)
        testing.assert_array_equal(
            orig_array.data.ptr, out_array.data.ptr)

    def test_stream(self):
        allowed_streams = ['null', True]
        if not cuda.runtime.is_hip:
            allowed_streams.append('ptds')

        # stream order is automatically established via DLPack protocol
        for src_s in [self._get_stream(s) for s in allowed_streams]:
            for dst_s in [self._get_stream(s) for s in allowed_streams]:
                with src_s:
                    orig_array = _gen_array(cupy.float32)
                    # If src_s != dst_s, dst_s waits until src_s complete.
                    # Null stream (0) must be passed as streamLegacy (1)
                    # on CUDA.
                    if not cuda.runtime.is_hip and dst_s.ptr == 0:
                        s_ptr = 1
                    else:
                        s_ptr = dst_s.ptr
                    dltensor = orig_array.__dlpack__(s_ptr)

                with dst_s:
                    out_array = cupy.from_dlpack(dltensor)
                    testing.assert_array_equal(orig_array, out_array)
                    testing.assert_array_equal(
                        orig_array.data.ptr, out_array.data.ptr)


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

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
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

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_multiple_consumption_error(self):
        # Prevent segfault, see #3611
        array = cupy.empty(10)
        tensor = array.toDlpack()
        array2 = cupy.fromDlpack(tensor)  # noqa
        with pytest.raises(ValueError) as e:
            array3 = cupy.fromDlpack(tensor)  # noqa
        assert 'consumed multiple times' in str(e.value)
