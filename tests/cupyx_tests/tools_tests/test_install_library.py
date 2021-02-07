import tempfile
import unittest
import urllib

from cupy import testing
from cupyx.tools import install_library


class TestInstallLibrary(unittest.TestCase):

    @testing.slow
    def test_install_cudnn(self):
        # Try installing cuDNN for all CUDA versions.
        for rec in install_library._cudnn_records:
            cuda = rec['cuda']
            with tempfile.TemporaryDirectory() as d:
                install_library.install_lib(cuda, d, 'cudnn')

    @testing.slow
    def test_install_cutensor(self):
        # Try installing cuTENSOR for all supported CUDA versions
        for rec in install_library._cutensor_records:
            cuda = rec['cuda']
            with tempfile.TemporaryDirectory() as d:
                install_library.install_lib(cuda, d, 'cutensor')

    def test_urls(self):
        assets = [r['assets'] for r in (
            install_library._cudnn_records
            + install_library._cutensor_records)]
        for asset in assets:
            for platform in asset.keys():
                url = asset[platform]['url']
                with urllib.request.urlopen(
                        urllib.request.Request(url, method='HEAD')) as resp:
                    assert resp.getcode() == 200

    def test_main(self):
        install_library.main(
            ['--library', 'cudnn', '--action', 'dump', '--cuda', 'null'])
        install_library.main(
            ['--library', 'cutensor', '--action', 'dump', '--cuda', 'null'])
