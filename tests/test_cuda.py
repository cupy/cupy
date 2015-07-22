import os
import re
import unittest

import numpy

from chainer import cuda
from chainer import testing


class TestCuda(unittest.TestCase):

    def _get_cuda_deps_requires(self):
        cwd = os.path.dirname(__file__)

        cuda_deps_path = os.path.join(cwd, '..', 'cuda_deps', 'setup.py')
        with open(cuda_deps_path) as f:
            in_require = False
            requires = []
            for line in f:
                if in_require:
                    if ']' in line:
                        in_require = False
                    else:
                        m = re.search(r'\'(.*)\',', line)
                        requires.append(m.group(1))
                else:
                    if 'install_requires' in line:
                        in_require = True

        return requires

    def test_requires(self):
        requires = self._get_cuda_deps_requires()
        self.assertSetEqual(set(['chainer'] + cuda._requires),
                            set(requires))

    def test_init_unavailable(self):
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                cuda.init()

    def test_to_gpu_unavailable(self):
        x = numpy.array([1])
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                cuda.to_gpu(x)

    def test_to_gpu_async_unavailable(self):
        x = numpy.array([1])
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                cuda.to_gpu_async(x)

    def test_empy_unavailable(self):
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                cuda.empty(())

    def test_empy_like_unavailable(self):
        x = numpy.array([1])
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                cuda.empty_like(x)


testing.run_module(__name__, __file__)
