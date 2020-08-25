import tempfile
import unittest

from cupy import testing
from cupyx.tools import install_library


class TestInstallLibrary(unittest.TestCase):

    @testing.slow
    def test_install_cudnn(self):
        # Try installing cuDNN for all CUDA versions.
        for rec in install_library._cudnn_records:
            cuda = rec['cuda']
            with tempfile.TemporaryDirectory() as d:
                install_library.install_cudnn(cuda, d)

    def test_main(self):
        install_library.main(
            ['--library', 'cudnn', '--action', 'dump', '--cuda', 'null'])
