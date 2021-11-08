import os
import pathlib
import subprocess
import sys
import shutil
import tempfile
import unittest

import numpy
import pytest

from cupy.cuda import nccl
from cupy import testing

from cupyx.distributed import init_process_group

nccl_available = nccl.available


N_WORKERS = 2


def _run_test(test_name, dtype=None):
    # subprocess is required not to interfere with cupy module imported in top
    # of this file
    temp_dir = tempfile.mkdtemp()
    try:
        script_path = os.path.join(temp_dir, 'script.py')
        template_path = pathlib.Path(__file__)
        with open(
                template_path.parent / 'test_comm_template.py.txt', 'r') as f:
            if dtype is None:
                dtype = ''
            else:
                dtype = '"{}"'.format(numpy.dtype(dtype).char)
            template = f.read().format(name=test_name, dtype=dtype)
        with open(script_path, 'w') as f:
            f.write(template)
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdoutdata, stderrdata = proc.communicate()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        assert stderrdata.decode() == ''
        assert proc.returncode == 0


@pytest.mark.skipif(not nccl_available, reason='nccl is not installed')
class TestNCCLBackend:
    @testing.for_all_dtypes(no_bool=True)
    def test_broadcast(self, dtype):
        _run_test('broadcast', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_reduce(self, dtype):
        _run_test('reduce', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_all_reduce(self, dtype):
        _run_test('all_reduce', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_reduce_scatter(self, dtype):
        _run_test('reduce_scatter', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_all_gather(self, dtype):
        _run_test('all_gather', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_send_and_recv(self, dtype):
        _run_test('send_and_recv', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_send_recv(self, dtype):
        _run_test('send_recv', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_scatter(self, dtype):
        _run_test('scatter', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_gather(self, dtype):
        _run_test('gather', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_all_to_all(self, dtype):
        _run_test('all_to_all', dtype)

    def test_barrier(self):
        _run_test('barrier')


@unittest.skipUnless(nccl_available, 'nccl is not installed')
class TestInitDistributed(unittest.TestCase):

    def test_init(self):
        _run_test('init')

    def test_invalid_backend(self):
        with pytest.raises(ValueError):
            init_process_group(1, 0, backend='mpi')

    def test_invalid_n_devices(self):
        with pytest.raises(ValueError):
            init_process_group(0, 0)

        with pytest.raises(ValueError):
            init_process_group(-1, 0)

    def test_invalid_rank(self):
        with pytest.raises(ValueError):
            init_process_group(2, -1)

        with pytest.raises(ValueError):
            init_process_group(2, 3)
