import re
import threading
import unittest

import pytest

import cupy
from cupy import cuda
from cupy.cuda import runtime
from cupy import testing


class TestDeviceComparison(unittest.TestCase):

    def check_eq(self, result, obj1, obj2):
        if result:
            assert obj1 == obj2
            assert obj2 == obj1
            assert not (obj1 != obj2)
            assert not (obj2 != obj1)
        else:
            assert obj1 != obj2
            assert obj2 != obj1
            assert not (obj1 == obj2)
            assert not (obj2 == obj1)

    def test_equality(self):
        self.check_eq(True, cuda.Device(0), cuda.Device(0))
        self.check_eq(True, cuda.Device(1), cuda.Device(1))
        self.check_eq(False, cuda.Device(0), cuda.Device(1))
        self.check_eq(False, cuda.Device(0), 0)
        self.check_eq(False, cuda.Device(0), None)
        self.check_eq(False, cuda.Device(0), object())

    def test_lt_device(self):
        assert cuda.Device(0) < cuda.Device(1)
        assert not (cuda.Device(0) < cuda.Device(0))
        assert not (cuda.Device(1) < cuda.Device(0))

    def test_le_device(self):
        assert cuda.Device(0) <= cuda.Device(1)
        assert cuda.Device(0) <= cuda.Device(0)
        assert not (cuda.Device(1) <= cuda.Device(0))

    def test_gt_device(self):
        assert not (cuda.Device(0) > cuda.Device(0))
        assert not (cuda.Device(0) > cuda.Device(0))
        assert cuda.Device(1) > cuda.Device(0)

    def test_ge_device(self):
        assert not (cuda.Device(0) >= cuda.Device(1))
        assert cuda.Device(0) >= cuda.Device(0)
        assert cuda.Device(1) >= cuda.Device(0)

    def check_comparison_other_type(self, obj1, obj2):
        with pytest.raises(TypeError):
            obj1 < obj2
        with pytest.raises(TypeError):
            obj1 <= obj2
        with pytest.raises(TypeError):
            obj1 > obj2
        with pytest.raises(TypeError):
            obj1 >= obj2
        with pytest.raises(TypeError):
            obj2 < obj1
        with pytest.raises(TypeError):
            obj2 <= obj1
        with pytest.raises(TypeError):
            obj2 > obj1
        with pytest.raises(TypeError):
            obj2 >= obj1

    def test_comparison_other_type(self):
        self.check_comparison_other_type(cuda.Device(0), 0)
        self.check_comparison_other_type(cuda.Device(0), 1)
        self.check_comparison_other_type(cuda.Device(1), 0)
        self.check_comparison_other_type(cuda.Device(1), None)
        self.check_comparison_other_type(cuda.Device(1), object())


@testing.gpu
class TestDeviceAttributes(unittest.TestCase):

    def test_device_attributes(self):
        d = cuda.Device()
        attributes = d.attributes
        assert isinstance(attributes, dict)
        assert all(isinstance(a, int) for a in attributes.values())
        # test a specific attribute that would be present on any supported GPU
        assert 'MaxThreadsPerBlock' in attributes

    def test_device_attributes_error(self):
        with pytest.raises(cuda.runtime.CUDARuntimeError):
            # try to retrieve attributes from a non-existent device
            cuda.device.Device(cuda.runtime.getDeviceCount()).attributes


@testing.gpu
class TestDevicePCIBusId(unittest.TestCase):
    def test_device_get_pci_bus_id(self):
        d = cuda.Device()
        pci_bus_id = d.pci_bus_id
        assert re.match(
            '^[a-fA-F0-9]{4}:[a-fA-F0-9]{2}:[a-fA-F0-9]{2}.[a-fA-F0-9]',
            pci_bus_id
        )

    def test_device_by_pci_bus_id(self):
        d1 = cuda.Device()
        d2 = cuda.Device.from_pci_bus_id(d1.pci_bus_id)
        assert d1 == d2
        d3 = cuda.Device(d2)
        assert d2 == d3

        with pytest.raises(cuda.runtime.CUDARuntimeError) as excinfo:
            cuda.Device.from_pci_bus_id('fake:id')
            assert excinfo == 'cudaErrorInvalidValue: invalid argument'

        with pytest.raises(cuda.runtime.CUDARuntimeError) as excinfo:
            cuda.Device.from_pci_bus_id('FFFF:FF:FF.F')
            assert excinfo == 'cudaErrorInvalidDevice: invalid device ordinal'


@testing.gpu
class TestDeviceHandles(unittest.TestCase):
    def _check_handle(self, func):
        handles = [func(), None, None]

        def _subthread():
            handles[1] = func()
            handles[2] = func()

        t = threading.Thread(target=_subthread)
        t.start()
        t.join()
        assert handles[0] is not None
        assert handles[0] != handles[1]
        assert handles[1] == handles[2]

    def test_cublas_handle(self):
        self._check_handle(cuda.get_cublas_handle)

    def test_cusolver_handle(self):
        self._check_handle(cuda.device.get_cusolver_handle)

    def test_cusolver_sp_handle(self):
        self._check_handle(cuda.device.get_cublas_handle)

    @pytest.mark.xfail(
        runtime.is_hip, reason='ROCm/HIP sparse support is not yet ready')
    def test_cusparse_handle(self):
        self._check_handle(cuda.device.get_cusparse_handle)


class TestDeviceFromPointer(unittest.TestCase):
    def test_from_pointer(self):
        assert cuda.device.from_pointer(cupy.empty(1).data.ptr).id == 0
