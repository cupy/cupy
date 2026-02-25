from __future__ import annotations

import collections
import os
import subprocess
import sys


# enable NEP 50 weak promotion rules
import numpy
if numpy.lib.NumpyVersion(numpy.__version__) < "2.0.0":
    numpy._set_promotion_state("weak")

# Enable `testdir` fixture to test `cupy.testing`.
# `pytest_plugins` cannot be locally configured. See also
# https://docs.pytest.org/en/stable/deprecations.html#pytest-plugins-in-non-top-level-conftest-files
pytest_plugins = ['pytester']


def _is_pip_installed():
    try:
        import pip  # NOQA
        return True
    except ImportError:
        return False


def _is_in_ci():
    ci_name = os.environ.get('CUPY_CI', '')
    return ci_name != ''


def pytest_configure(config):
    # Print installed packages
    if _is_in_ci() and _is_pip_installed():
        print("***** Installed packages *****", flush=True)
        subprocess.check_call([sys.executable, '-m', 'pip', 'freeze', '--all'])

    if config.pluginmanager.hasplugin("xdist"):
        n_gpu = os.environ.get('CUPY_TEST_GPU_LIMIT')
        worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'master')

        if n_gpu is None:
            if worker_id == 'master':
                print('\nTIP: when using pytest-xdist, you can automatically '
                      'rotate CUDA_VISIBLE_DEVICES for each test worker by '
                      'setting CUPY_TEST_GPU_LIMIT environment variable.\n')
        else:
            # We'll edit the environment variable `CUDA_VISIBLE_DEVICES` for
            # each worker which needs to happen before the CUDA init.
            n_gpu = int(n_gpu)

            devices = os.environ.get('CUDA_VISIBLE_DEVICES')
            if devices is None:
                devices = [str(k) for k in range(n_gpu)]
            else:
                devices = devices.split(',')[:n_gpu]

            if worker_id == 'master':
                print(f'\nNOTE: Setting workers to use a shifted version of:'
                      f'\n    CUDA_VISIBLE_DEVICES={",".join(devices)}\n')
            else:
                assert worker_id.startswith('gw')
                w = int(worker_id[2:])

                devices = collections.deque(devices)
                # left rotate so second worker gets second GPU
                devices.rotate(-w)
                devices = ','.join(devices)
                os.environ['CUDA_VISIBLE_DEVICES'] = devices


if int(os.environ.get('CUPY_ENABLE_UMP', 0)) != 0:
    # Make sure malloc is used in a stream-ordered fashion
    import cupy as cp
    cp.cuda.set_allocator(cp.cuda.MemoryPool(
        cp.cuda.memory.malloc_system).malloc)

    import cupy._core.numpy_allocator as ac
    import numpy_allocator
    import ctypes
    lib = ctypes.CDLL(ac.__file__)

    class my_allocator(metaclass=numpy_allocator.type):
        _calloc_ = ctypes.addressof(lib._calloc)
        _malloc_ = ctypes.addressof(lib._malloc)
        _realloc_ = ctypes.addressof(lib._realloc)
        _free_ = ctypes.addressof(lib._free)
    my_allocator.__enter__()
