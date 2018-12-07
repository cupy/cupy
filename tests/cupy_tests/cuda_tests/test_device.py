import unittest

import pytest
import six

from cupy import cuda
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

    @unittest.skipIf(
        six.PY2, 'Python 2 comparison result of objects is arbitrary')
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
