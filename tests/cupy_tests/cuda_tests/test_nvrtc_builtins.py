import os
import shutil
import sys
import unittest

from cupy.cuda import nvrtc_builtins
from cupy import testing


config = nvrtc_builtins._nvrtc_platform_config.get(sys.platform, None)


@unittest.skipIf(config is None, 'unsupported environment')
class TestNvrtcBuiltinsCheck(unittest.TestCase):

    def test_get_nvrtc_path(self):
        path = nvrtc_builtins._get_nvrtc_path()
        assert path is not None
        assert os.path.lexists(path)

    def test_get_nvrtc_builtins_path(self):
        path = nvrtc_builtins._get_nvrtc_builtins_path(
            config['nvrtc-builtins'])
        assert path is not None

    def test_check(self):
        with testing.assert_no_warns():
            nvrtc_builtins.check()

    def test_check_fail_missing(self):
        # Test for missing nvrtc-builtins library.
        libpath = config['nvrtc-builtins']
        assert not os.path.lexists(libpath)

        default_builtins = config['nvrtc-builtins']
        try:
            # Create an empty file to cause load failure.
            with open(libpath, 'w'):
                pass

            config['nvrtc-builtins'] = './{}'.format(default_builtins)
            with testing.assert_warns(UserWarning) as w:
                nvrtc_builtins.check()
            assert len(w) == 1
            assert 'could not be loaded' in str(w[0].message)
        finally:
            config['nvrtc-builtins'] = default_builtins
            if os.path.lexists(libpath):
                os.remove(libpath)

    def test_check_fail_version_mismatch(self):
        # Test for version mismatch of nvrtc-builtins library.
        libpath = config['nvrtc-builtins']
        librealpath = nvrtc_builtins._get_nvrtc_builtins_path(
            config['nvrtc-builtins'])
        libfakepath = '{}.0.0'.format(libpath)

        assert not os.path.lexists(libpath)
        assert not os.path.lexists(libfakepath)

        default_builtins = config['nvrtc-builtins']
        try:
            # Copy real library (e.g. `libnvrtc-builtins.so.9.0.176` with
            # fake name (append `.0`).
            # Then symlink it to `libnvrtc-builtins.so`.
            shutil.copy(librealpath, libfakepath)
            os.symlink(libfakepath, libpath)

            config['nvrtc-builtins'] = './{}'.format(default_builtins)
            with testing.assert_warns(UserWarning) as w:
                nvrtc_builtins.check()
            assert len(w) == 1
            assert 'Version mismatch' in str(w[0].message)
        finally:
            config['nvrtc-builtins'] = default_builtins
            if os.path.lexists(libfakepath):
                os.remove(libfakepath)
            if os.path.lexists(libpath):
                os.remove(libpath)
