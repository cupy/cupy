import os
import platform
import tempfile
import unittest
import urllib

from cupy import testing
from cupyx.tools import install_library

import pytest


@testing.parameterize(
    {'library': 'cudnn'},
    {'library': 'cutensor'},
    {'library': 'nccl'},
)
class TestInstallLibrary(unittest.TestCase):

    @testing.slow
    def test_install(self):
        system = platform.system()
        if system == 'Windows' and self.library == 'nccl':
            pytest.skip('NCCL is only available for Linux')

        # Try installing library for all supported CUDA versions
        for rec in install_library.library_records[self.library]:
            cuda = rec['cuda']
            version = rec[self.library]
            filename = rec['assets'][system]['filename']
            with tempfile.TemporaryDirectory() as d:
                install_library.install_lib(cuda, d, self.library)
                self._check_installed(d, cuda, self.library, version, filename)

    def _check_installed(self, prefix, cuda, lib, version, filename):
        install_root = os.path.join(prefix, cuda, lib, version)
        assert os.path.isdir(install_root)
        for _x, _y, files in os.walk(install_root):
            if filename in files:
                return
        pytest.fail('expected file cound not be found')

    def test_urls(self):
        assets = [r['assets']
                  for r in install_library.library_records[self.library]]
        for asset in assets:
            for system in asset.keys():
                url = asset[system]['url']
                with urllib.request.urlopen(
                        urllib.request.Request(url, method='HEAD')) as resp:
                    assert resp.getcode() == 200

    def test_main(self):
        install_library.main(
            ['--library', self.library, '--action', 'dump', '--cuda', 'null'])
