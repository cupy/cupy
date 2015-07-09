import os
import re
import unittest

from chainer import cuda

if cuda.available:
    cuda.init()


class TestCuda(unittest.TestCase):

    def test_requires(self):
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

            self.assertSetEqual(set(['chainer'] + cuda._requires),
                                set(requires))
