import os
import platform
import tempfile
import urllib

from cupy import testing
from cupyx.tools import install_library

import pytest


_libraries = ['cudnn', 'nccl', 'cutensor']


def _get_supported_cuda_versions(lib):
    return sorted(set([
        rec['cuda'] for rec in install_library.library_records[lib]]))


class TestInstallLibrary:

    @pytest.mark.parametrize('cuda', _get_supported_cuda_versions('cudnn'))
    @testing.slow
    def test_install_cudnn(self, cuda):
        self._test_install('cudnn', cuda)

    @pytest.mark.skipif(
        platform.system() == 'Windows',
        reason='NCCL is only available for Linux')
    @pytest.mark.parametrize('cuda', _get_supported_cuda_versions('nccl'))
    @testing.slow
    def test_install_nccl(self, cuda):
        self._test_install('nccl', cuda)

    @pytest.mark.parametrize('cuda', _get_supported_cuda_versions('cutensor'))
    @testing.slow
    def test_install_cutensor(self, cuda):
        self._test_install('cutensor', cuda)

    def _test_install(self, library, cuda):
        system = platform.system()
        for rec in install_library.library_records[library]:
            if rec['cuda'] != cuda:
                continue
            version = rec[library]
            filenames = rec['assets'][system]['filenames']
            with tempfile.TemporaryDirectory() as d:
                install_library.install_lib(cuda, d, library)
                self._check_installed(
                    d, cuda, library, version, filenames)
            break
        else:
            pytest.fail(f'unexpected CUDA version {cuda} for {library}')

    def _check_installed(self, prefix, cuda, lib, version, filenames):
        install_root = os.path.join(prefix, cuda, lib, version)
        assert os.path.isdir(install_root)
        for _x, _y, files in os.walk(install_root):
            for filename in filenames:
                if filename in files:
                    return
        pytest.fail('expected file cound not be found')

    @pytest.mark.parametrize('library', _libraries)
    def test_urls(self, library):
        assets = [r['assets']
                  for r in install_library.library_records[library]]
        for asset in assets:
            for system in asset.keys():
                url = asset[system]['url']
                with urllib.request.urlopen(
                        urllib.request.Request(url, method='HEAD')) as resp:
                    assert resp.getcode() == 200

    @pytest.mark.parametrize('library', _libraries)
    def test_main(self, library):
        install_library.main(
            ['--library', library, '--action', 'dump', '--cuda', 'null'])
