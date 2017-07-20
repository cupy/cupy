import nose
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy

import cupy
from cupy import testing


def _run_script(code):
    # subprocess is required not to interfere with cupy module imported in top
    # of this file
    temp_dir = tempfile.mkdtemp()
    try:
        script_path = os.path.join(temp_dir, 'script.py')
        with open(script_path, 'w') as f:
            f.write(code)
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdoutdata, stderrdata = proc.communicate()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return proc.returncode, stdoutdata, stderrdata


def _test_cupy_available(self):
    returncode, stdoutdata, stderrdata = _run_script('''
import cupy
print(cupy.is_available())''')
    self.assertEqual(returncode, 0, 'stderr: {!r}'.format(stderrdata))
    self.assertIn(stdoutdata, (b'True\n', b'False\n'))
    return stdoutdata == b'True\n'


class TestImportError(unittest.TestCase):

    def test_import_error(self):
        returncode, stdoutdata, stderrdata = _run_script('''
try:
    import cupy
except Exception as e:
    print(type(e).__name__)
''')
        self.assertEqual(returncode, 0, 'stderr: {!r}'.format(stderrdata))
        self.assertIn(stdoutdata, (b'', b'RuntimeError\n'))


class TestAvailable(unittest.TestCase):

    @testing.gpu
    def test_available(self):
        available = _test_cupy_available(self)
        self.assertTrue(available)


class TestNotAvailable(unittest.TestCase):

    def setUp(self):
        self.old = os.environ.get('CUDA_VISIBLE_DEVICES')

    def tearDown(self):
        if self.old is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.old

    def test_no_device_1(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        available = _test_cupy_available(self)
        self.assertFalse(available)

    def test_no_device_2(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        available = _test_cupy_available(self)
        self.assertFalse(available)


class TestNumpyShortcuts(unittest.TestCase):

    def test_nan(self):
        self.assertIs(cupy.nan, numpy.nan)

    def test_check_nan_with_isnan(self):
        nan_array = cupy.full((3,), cupy.nan)
        self.assertTrue(cupy.isnan(nan_array).all())

    def test_check_nan_with_isnan_2(self):
        self.assertTrue(cupy.isnan(cupy.nan).all())

# This is copied from chainer/testing/__init__.py, so should be replaced in
# some way.
if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
