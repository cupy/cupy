import pathlib
import subprocess
import sys
import unittest

import numpy
import pytest

from cupy.cuda import nccl
from cupy import testing

from cupyx.distributed import init_process_group

nccl_available = nccl.available


def _run_test(test_name, dtype=None):
    # subprocess is required not to interfere with cupy module imported in top
    # of this file
    runner_path = pathlib.Path(__file__).parent / 'comm_runner.py'
    args = [sys.executable, runner_path, test_name]
    if dtype is not None:
        args.append(numpy.dtype(dtype).char)
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdoutdata, stderrdata = proc.communicate()
    assert stderrdata.decode() == ''
    assert proc.returncode == 0


@pytest.mark.skipif(not nccl_available, reason='nccl is not installed')
@testing.multi_gpu(2)
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


@pytest.mark.skipif(not nccl_available, reason='nccl is not installed')
class TestInitDistributed(unittest.TestCase):

    @testing.multi_gpu(2)
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
