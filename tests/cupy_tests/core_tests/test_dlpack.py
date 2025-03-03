import numpy
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
    elif dtype == cupy.bool_:
        array = cupy.random.randint(0, 2, size=(2, 3)).astype(cupy.bool_)
    else:
        assert False, f'unrecognized dtype: {dtype}'
    return array


class DLDummy:
    """Dummy object to wrap a __dlpack__ capsule, so we can use from_dlpack.
    """

    def __init__(self, capsule, device):
        self.capsule = capsule
        self.device = device

    def __dlpack__(self, *args, **kwargs):
        return self.capsule

    def __dlpack_device__(self):
        return self.device


class TestDLPackConversion:

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    @testing.for_all_dtypes(no_bool=False)
    def test_conversion(self, dtype):
        orig_array = _gen_array(dtype)
        tensor = orig_array.toDlpack()
        assert '"dltensor"' in repr(tensor)  # unversioned one

        out_array = cupy.fromDlpack(tensor)
        testing.assert_array_equal(orig_array, out_array)
        testing.assert_array_equal(orig_array.data.ptr, out_array.data.ptr)


class TestNewDLPackConversion:

    @pytest.fixture(autouse=True, params=["device", "managed"])
    def pool(self, request):
        self.memory = request.param
        if self.memory == 'managed':
            if cuda.runtime.is_hip:
                pytest.skip('HIP does not support managed memory')
            old_pool = cupy.get_default_memory_pool()
            new_pool = cuda.MemoryPool(cuda.malloc_managed)
            cuda.set_allocator(new_pool.malloc)

            yield

            cuda.set_allocator(old_pool.malloc)
        else:
            # Nothing to do, we can use the default pool.
            yield

        del self.memory

    def _get_stream(self, stream_name):
        if stream_name == 'null':
            return cuda.Stream.null
        elif stream_name == 'ptds':
            return cuda.Stream.ptds
        else:
            return cuda.Stream()

    @testing.for_all_dtypes(no_bool=False)
    def test_conversion(self, dtype):
        orig_array = _gen_array(dtype)
        out_array = cupy.from_dlpack(orig_array)
        testing.assert_array_equal(orig_array, out_array)
        testing.assert_array_equal(
            orig_array.data.ptr, out_array.data.ptr)

    def test_from_dlpack_and_conv_errors(self):
        orig_array = _gen_array("int8")

        with pytest.raises(NotImplementedError):
            cupy.from_dlpack(orig_array, device=orig_array.device)

        with pytest.raises(BufferError):
            # Currently CuPy's `__dlpack__` only allows `copy=True`
            # for host copies.
            cupy.from_dlpack(orig_array, copy=True)

    @pytest.mark.parametrize("kwargs, versioned", [
        ({}, False), ({"max_version": None}, False),
        ({"max_version": (1, 0)}, True), ({"max_version": (10, 10)}, True),
        ({"max_version": (0, 8)}, False)
    ])
    def test_conversion_max_version(self, kwargs, versioned):
        orig_array = _gen_array("int8")

        capsule = orig_array.__dlpack__(**kwargs)
        # We can identify if the version is correct via the name:
        if versioned:
            assert '"dltensor_versioned"' in str(capsule)
        else:
            assert '"dltensor"' in str(capsule)

        out_array = cupy.from_dlpack(
            DLDummy(capsule, orig_array.__dlpack_device__()))

        testing.assert_array_equal(orig_array, out_array)
        testing.assert_array_equal(
            orig_array.data.ptr, out_array.data.ptr)

    def test_conversion_device(self):
        orig_array = _gen_array("float32")

        # If the device is identical, then we support it:
        capsule = orig_array.__dlpack__(
            dl_device=orig_array.__dlpack_device__())
        out_array = cupy.from_dlpack(
            DLDummy(capsule, orig_array.__dlpack_device__()))

        testing.assert_array_equal(orig_array, out_array)
        testing.assert_array_equal(
            orig_array.data.ptr, out_array.data.ptr)

    def test_conversion_bad_device(self):
        arr = _gen_array("float32")

        # invalid device ID
        with pytest.raises(BufferError):
            arr.__dlpack__(dl_device=(arr.__dlpack_device__()[0], 2**30))

        # Simple, non-matching device:
        with pytest.raises(BufferError):
            arr.__dlpack__(dl_device=(9, 0))

    @testing.with_requires('numpy>=1.23.0')
    def test_conversion_device_to_cpu(self):
        # NOTE: This defaults to the old unversioned, which is needed for
        #       NumPy 1.x support.
        # If (and only if) the device is managed, we also support exporting
        # to CPU.
        orig_array = _gen_array("float32")

        arr1 = numpy.from_dlpack(
            DLDummy(orig_array.__dlpack__(dl_device=(1, 0)), device=(1, 0)))
        arr2 = numpy.from_dlpack(
            DLDummy(orig_array.__dlpack__(dl_device=(1, 0)), device=(1, 0)))

        numpy.testing.assert_array_equal(orig_array.get(), arr1)
        assert orig_array.dtype == arr1.dtype
        # Arrays share the same memory exactly when memory is managed.
        assert numpy.may_share_memory(arr1, arr2) == (self.memory == "managed")

        arr_copy = numpy.from_dlpack(DLDummy(
            orig_array.__dlpack__(dl_device=(1, 0), copy=True), device=(1, 0)))
        # The memory must not be shared with with a copy=True request
        assert not numpy.may_share_memory(arr_copy, arr1)
        numpy.testing.assert_array_equal(arr1, arr_copy)

        # Also test copy=False
        if self.memory != "managed":
            with pytest.raises(ValueError):
                orig_array.__dlpack__(dl_device=(1, 0), copy=False)
        else:
            arr_nocopy = numpy.from_dlpack(DLDummy(
                orig_array.__dlpack__(dl_device=(1, 0), copy=False),
                device=(1, 0))
            )
            assert numpy.may_share_memory(arr_nocopy, arr1)

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
                    dltensor = orig_array.__dlpack__(stream=s_ptr)

                with dst_s:
                    out_array = cupy.from_dlpack(dltensor)
                    testing.assert_array_equal(orig_array, out_array)
                    testing.assert_array_equal(
                        orig_array.data.ptr, out_array.data.ptr)


class TestDLTensorMemory:

    @pytest.fixture
    def pool(self):
        old_pool = cupy.get_default_memory_pool()
        pool = cupy.cuda.MemoryPool()
        cupy.cuda.set_allocator(pool.malloc)

        yield pool

        pool.free_all_blocks()
        cupy.cuda.set_allocator(old_pool.malloc)

    @pytest.mark.parametrize('max_version', [None, (1, 0)])
    def test_deleter(self, pool, max_version):
        # memory is freed when tensor is deleted, as it's not consumed
        array = cupy.empty(10)
        tensor = array.__dlpack__(max_version=max_version)
        # str(tensor): <capsule object "dltensor" at 0x7f7c4c835330>
        name = "dltensor" if max_version is None else "dltensor_versioned"
        assert f'"{name}"' in str(tensor)
        assert pool.n_free_blocks() == 0
        del array
        assert pool.n_free_blocks() == 0
        del tensor
        assert pool.n_free_blocks() == 1

    @pytest.mark.parametrize('max_version', [None, (1, 0)])
    def test_deleter2(self, pool, max_version):
        # memory is freed when array2 is deleted, as tensor is consumed
        array = cupy.empty(10)
        tensor = array.__dlpack__(max_version=max_version)
        name = "dltensor" if max_version is None else "dltensor_versioned"
        assert f'"{name}"' in str(tensor)
        array2 = cupy.from_dlpack(
            DLDummy(tensor, device=array.__dlpack_device__()))
        assert f'"used_{name}"' in str(tensor)
        assert pool.n_free_blocks() == 0
        del array
        assert pool.n_free_blocks() == 0
        del array2
        assert pool.n_free_blocks() == 1
        del tensor
        assert pool.n_free_blocks() == 1

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_multiple_consumption_error(self):
        # Prevent segfault, see #3611
        array = cupy.empty(10)
        tensor = array.toDlpack()
        array2 = cupy.fromDlpack(tensor)  # noqa
        with pytest.raises(ValueError) as e:
            array3 = cupy.fromDlpack(tensor)  # noqa
        assert 'consumed multiple times' in str(e.value)
