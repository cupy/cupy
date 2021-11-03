import filecmp
import os
import tempfile
import unittest

import numpy

from cupy import testing
import cupy


@testing.gpu
class TestText(unittest.TestCase):

    def test_savetxt(self):
        tmp_cupy = tempfile.NamedTemporaryFile(delete=False)
        tmp_numpy = tempfile.NamedTemporaryFile(delete=False)
        try:
            tmp_cupy.close()
            tmp_numpy.close()
            array = [[1, 2, 3], [2, 3, 4]]
            cupy.savetxt(tmp_cupy.name, cupy.array(array))
            numpy.savetxt(tmp_numpy.name, numpy.array(array))
            assert filecmp.cmp(tmp_cupy.name, tmp_numpy.name)
        finally:
            os.remove(tmp_cupy.name)
            os.remove(tmp_numpy.name)
