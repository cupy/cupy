import collections
import os

import pytest


# Enable `testdir` fixture to test `cupy.testing`.
# `pytest_plugins` cannot be locally configured. See also
# https://docs.pytest.org/en/stable/deprecations.html#pytest-plugins-in-non-top-level-conftest-files
pytest_plugins = ['pytester']


def pytest_configure(config):
    if config.pluginmanager.hasplugin("xdist"):
        config.pluginmanager.register(DeferPlugin())


# https://docs.pytest.org/en/latest/how-to/writing_hook_functions.html#optionally-using-hooks-from-3rd-party-plugins
class DeferPlugin:
    """Simple plugin to defer pytest-xdist hook functions."""

    # Edit the environment variable `CUDA_VISIBLE_DEVICES` for each session.
    # Cannot use `pytest_configure_node` nor `pytest_testnodeready` hook,
    # because they are called in the `master` node (process).
    # See also https://github.com/pytest-dev/pytest-xdist/issues/179.
    @pytest.fixture(autouse=True, scope='session')
    def _rotate_cuda_visible_devices(self, worker_id):
        if worker_id == 'master':
            # `worker_id` can be `master` if `pytest-xdist` is installed and
            # run without `-n` option.
            return

        n_gpu = os.environ.get('CUPY_TEST_GPU_LIMIT')
        if n_gpu is None:
            print('Tip: when using pytest-xdist, you can automatically rotate'
                  ' CUDA_VISIBLE_DEVICES for each test worker by setting'
                  ' CUPY_TEST_GPU_LIMIT environment variable.')
            return
        n_gpu = int(n_gpu)

        assert worker_id.startswith('gw')
        w = int(worker_id[2:])

        devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if devices is None:
            devices = [str(k) for k in range(n_gpu)]
        else:
            devices = devices.split(',')[:n_gpu]
        devices = collections.deque(devices)
        devices.rotate(w)
        devices = ','.join(devices)
        os.environ['CUDA_VISIBLE_DEVICES'] = devices
        # With PyTest's default, the print will be shown as
        # "--- Captured stdout setup ---" on failure.
        print(f'CUDA_VISIBLE_DEVICES={devices}')
