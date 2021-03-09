import os
import platform
import tempfile
import unittest
import urllib

from cupy import testing
from cupyx.tools import install_library

import pytest


_platform_name = platform.system()


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

    @testing.slow
    @pytest.mark.skipif(
        _platform_name == 'Windows', reason='NCCL is only available for Linux')
    def test_install_nccl(self):
        # Try installing NCCL for all supported CUDA versions
        for rec in install_library._nccl_records:
            cuda = rec['cuda']
            version = rec['nccl']
            filename = rec['assets'][_platform_name]['filename']
            with tempfile.TemporaryDirectory() as d:
                install_library.install_lib(cuda, d, 'nccl')
                self._check_file_exists(d, cuda, 'nccl', version, filename)

    def _check_file_exists(self, prefix, cuda, lib, version, filename):
        install_root = os.path.join(prefix, cuda, lib, version)
        assert os.path.isdir(install_root)
        for _x, _y, files in os.walk(install_root):
            print(_x, _y, files)
            if filename in files:
                return
        pytest.fail('expected file cound not be found')

    def test_urls(self):
        assets = [r['assets'] for r in (
            install_library._cudnn_records
            + install_library._cutensor_records
            + install_library._nccl_records)]
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
        install_library.main(
            ['--library', 'nccl', '--action', 'dump', '--cuda', 'null'])
