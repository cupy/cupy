import unittest

import six

from cupy.cuda import compiler
from cupy import testing


@testing.gpu
class TestNvrtcStderr(unittest.TestCase):

    def test(self):
        # An error message contains the file name `kern.cu`
        with six.assertRaisesRegex(self, compiler.CompileException, 'kern.cu'):
            compiler.compile_using_nvrtc('a')
