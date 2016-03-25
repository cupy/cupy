import distutils
import setuptools  # NOQA
import unittest

from install import build


class TestCheckVersion(unittest.TestCase):

    def setUp(self):
        self.compiler = distutils.ccompiler.new_compiler()
        self.settings = {'include_dirs': []}

    def test_check_cuda_version(self):
        self.assertTrue(build.check_cuda_version(
            self.compiler, self.settings))

    def test_check_cudnn_version(self):
        self.assertTrue(build.check_cudnn_version(
            self.compiler, self.settings))
