import pathlib
import subprocess
import sys
import unittest
import os

import numpy
import pytest

from cupy.cuda import nccl
from cupy import testing

from cupyx.distributed import init_process_group
from cupyx.distributed._nccl_comm import _mpi_available


nccl_available = nccl.available


def _run_test(test_name, dtype=None):
    # subprocess is required not to interfere with cupy module imported in top
    # of this file
    runner_path = pathlib.Path(__file__).parent / 'comm_runner.py'
    args = [sys.executable, runner_path, test_name, 'store']
    if dtype is not None:
        args.append(numpy.dtype(dtype).char)
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdoutdata, stderrdata = proc.communicate()
    assert stderrdata.decode() == ''
    assert proc.returncode == 0


def _run_test_with_mpi(test_name, dtype=None):
    # subprocess is required not to interfere with cupy module imported in top
    # of this file
    runner_path = pathlib.Path(__file__).parent / 'comm_runner.py'
    args = ['mpiexec', '-n', '2', '--allow-run-as-root',
            sys.executable, runner_path, test_name, 'mpi']
    if dtype is not None:
        args.append(numpy.dtype(dtype).char)
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ
    )
    stdoutdata, stderrdata = proc.communicate()
    assert stderrdata.decode() == ''
    assert proc.returncode == 0


@pytest.mark.skipif(not nccl_available, reason='nccl is not installed')
@testing.multi_gpu(2)
class TestNCCLBackend:
    def _run_test(self, test, dtype):
        _run_test(test, dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_broadcast(self, dtype):
        self._run_test('broadcast', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_reduce(self, dtype):
        self._run_test('reduce', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_all_reduce(self, dtype):
        self._run_test('all_reduce', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_reduce_scatter(self, dtype):
        self._run_test('reduce_scatter', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_all_gather(self, dtype):
        self._run_test('all_gather', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_send_and_recv(self, dtype):
        self._run_test('send_and_recv', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_send_recv(self, dtype):
        self._run_test('send_recv', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_scatter(self, dtype):
        self._run_test('scatter', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_gather(self, dtype):
        self._run_test('gather', dtype)

    @testing.for_all_dtypes(no_bool=True)
    def test_all_to_all(self, dtype):
        self._run_test('all_to_all', dtype)

    def test_barrier(self):
        self._run_test('barrier', None)


@pytest.mark.skipif(not _mpi_available, reason='mpi is not installed')
@testing.multi_gpu(2)
class TestNCCLBackendWithMPI(TestNCCLBackend):
    def _run_test(self, test, dtype):
        _run_test_with_mpi(test, dtype)


@pytest.mark.skipif(not nccl_available, reason='nccl is not installed')
@testing.multi_gpu(2)
class TestNCCLBackendSparse:
    def _run_test(self, test, dtype):
        _run_test(test, dtype)

    @testing.for_dtypes('fdFD')
    def test_send_and_recv(self, dtype):
        self._run_test('sparse_send_and_recv', dtype)

    @testing.for_dtypes('fdFD')
    def test_broadcast(self, dtype):
        self._run_test('sparse_broadcast', dtype)

    @testing.for_dtypes('fdFD')
    def test_reduce(self, dtype):
        self._run_test('sparse_reduce', dtype)

    @testing.for_dtypes('fdFD')
    def test_all_reduce(self, dtype):
        self._run_test('sparse_all_reduce', dtype)

    @testing.for_dtypes('fdFD')
    def test_scatter(self, dtype):
        self._run_test('sparse_scatter', dtype)

    @testing.for_dtypes('fdFD')
    def test_gather(self, dtype):
        self._run_test('sparse_gather', dtype)

    @testing.for_dtypes('fdFD')
    def test_all_gather(self, dtype):
        self._run_test('sparse_all_gather', dtype)

    @testing.for_dtypes('fdFD')
    def test_all_to_all(self, dtype):
        self._run_test('sparse_all_to_all', dtype)

    @testing.for_dtypes('fdFD')
    def test_reduce_scatter(self, dtype):
        self._run_test('sparse_reduce_scatter', dtype)

    @testing.for_dtypes('fdFD')
    def test_send_recv(self, dtype):
        self._run_test('sparse_send_recv', dtype)


@pytest.mark.skipif(not _mpi_available, reason='mpi is not installed')
@testing.multi_gpu(2)
class TestNCCLBackendSparseWithMPI(TestNCCLBackendSparse):
    def _run_test(self, test, dtype):
        _run_test_with_mpi(test, dtype)


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
