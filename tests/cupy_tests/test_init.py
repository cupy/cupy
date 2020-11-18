import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy

import cupy
from cupy import testing
import cupyx


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
    assert returncode == 0, 'stderr: {!r}'.format(stderrdata)
    assert stdoutdata in (b'True\n', b'True\r\n', b'False\n', b'False\r\n')
    return stdoutdata == b'True\n' or stdoutdata == b'True\r\n'


class TestImportError(unittest.TestCase):

    def test_import_error(self):
        returncode, stdoutdata, stderrdata = _run_script('''
try:
    import cupy
except Exception as e:
    print(type(e).__name__)
''')
        assert returncode == 0, 'stderr: {!r}'.format(stderrdata)
        assert stdoutdata in (b'', b'RuntimeError\n')


if not cupy.cuda.runtime.is_hip:
    visible = 'CUDA_VISIBLE_DEVICES'
else:
    visible = 'HIP_VISIBLE_DEVICES'


class TestAvailable(unittest.TestCase):

    @testing.gpu
    def test_available(self):
        available = _test_cupy_available(self)
        assert available


class TestNotAvailable(unittest.TestCase):

    def setUp(self):
        self.old = os.environ.get(visible)

    def tearDown(self):
        if self.old is None:
            os.environ.pop(visible)
        else:
            os.environ[visible] = self.old

    @unittest.skipIf(cupy.cuda.runtime.is_hip,
                     'HIP handles empty HIP_VISIBLE_DEVICES differently')
    def test_no_device_1(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = ' '
        available = _test_cupy_available(self)
        assert not available

    def test_no_device_2(self):
        os.environ[visible] = '-1'
        available = _test_cupy_available(self)
        assert not available


class TestMemoryPool(unittest.TestCase):

    def test_get_default_memory_pool(self):
        p = cupy.get_default_memory_pool()
        assert isinstance(p, cupy.cuda.memory.MemoryPool)

    def test_get_default_pinned_memory_pool(self):
        p = cupy.get_default_pinned_memory_pool()
        assert isinstance(p, cupy.cuda.pinned_memory.PinnedMemoryPool)


class TestShowConfig(unittest.TestCase):

    def test_show_config(self):
        with mock.patch('sys.stdout.write') as write_func:
            cupy.show_config()
        write_func.assert_called_once_with(str(cupyx.get_runtime_info()))


class TestAliases(unittest.TestCase):

    def test_abs_is_absolute(self):
        for xp in (numpy, cupy):
            assert xp.abs is xp.absolute

    def test_conj_is_conjugate(self):
        for xp in (numpy, cupy):
            assert xp.conj is xp.conjugate

    def test_bitwise_not_is_invert(self):
        for xp in (numpy, cupy):
            assert xp.bitwise_not is xp.invert


# This is copied from chainer/testing/__init__.py, so should be replaced in
# some way.
if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-vvs', '-x', '--pdb'])
