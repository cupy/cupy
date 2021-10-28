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
        with (tempfile.NamedTemporaryFile() as tmp_cupy,
              tempfile.NamedTemporaryFile() as tmp_numpy):
            array = [[1, 2, 3], [2, 3, 4]]
            cupy.savetxt(tmp_cupy.name, cupy.array(array))
            numpy.savetxt(tmp_numpy.name, numpy.array(array))
            assert filecmp.cmp(tmp_cupy.name, tmp_numpy.name)
