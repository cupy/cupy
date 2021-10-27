import unittest
import tempfile
import filecmp
import os


from cupy import testing
import cupy
import numpy


@testing.gpu
class TestText(unittest.TestCase):
        
    def test_savetxt(self):
        array = [[1, 2, 3], [2, 3, 4]]
        tmp_cupy = tempfile.NamedTemporaryFile('w+t')
        tmp_cupy.name = "test_cupy.txt"
        tmp_numpy = tempfile.NamedTemporaryFile('w+t')
        tmp_numpy.name = "test_numpy.txt"
        cupy.savetxt(tmp_cupy.name, cupy.array(array))
        numpy.savetxt(tmp_numpy.name, numpy.array(array))
        assert filecmp.cmp(tmp_cupy.name, tmp_numpy.name) is True
        os.remove(str(tmp_cupy.name))
        os.remove(str(tmp_numpy.name))
    pass
