from __future__ import annotations

import collections
import os
import subprocess
import sys

import pytest


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


# ---------------------------------------------------------------------------
# Ascend (NPU) 不支持的 dtype: 配置 + 收集期 skip
#
# NPU 只保证 float32 (float64 仅四则运算可用, complex 无算子), 若不处理,
# 上游 CuPy 测试里大量 dtype 参数化用例会直接 FAIL, 淹没真实的移植缺陷。
# 过滤策略集中在 `cupy.testing._ascend_dtypes` (默认 auto: 仅 Ascend 生效,
# 跳过 float64/complex64/complex128), 这里负责:
#   1. 暴露 pytest 配置 (--ascend-dtype-filter / ini ascend_dtype_filter);
#   2. 把 `@pytest.mark.parametrize('dtype', [...])` 这类**显式**参数化用例
#      标记为 skip (decorator 型参数化由 cupy.testing._loops 在源头过滤);
#   3. 在 header 里打印生效的过滤策略, 避免"为什么没跑 float64"的困惑。
# ---------------------------------------------------------------------------
def pytest_addoption(parser):
    group = parser.getgroup('ascend')
    group.addoption(
        '--ascend-dtype-filter', action='store', default=None,
        choices=['auto', 'on', 'off'],
        help='Ascend 不支持的 dtype (float64/complex*) 是否从测试参数化中去掉; '
             'auto=按后端自动判断 (仅 Ascend 生效)。')
    parser.addini(
        'ascend_dtype_filter', 'auto | on | off (Ascend dtype 过滤模式)',
        default='auto')


def _ascend_dtypes():
    from cupy.testing import _ascend_dtypes as module
    return module


def _configure_ascend_dtype_filter(config):
    """把 pytest 侧配置同步给 cupy.testing (供 _loops 的 import 期过滤使用)。"""
    mode = None
    try:
        mode = config.getoption('ascend_dtype_filter')
    except (ValueError, KeyError):
        pass
    if mode is None:
        try:
            mode = config.getini('ascend_dtype_filter')
        except (ValueError, KeyError):
            mode = None
    try:
        _ascend_dtypes().configure(mode)
    except Exception as e:  # pragma: no cover - 配置错误不应让 pytest 崩掉
        print(f'warning: 无法应用 ascend dtype filter 配置 ({mode!r}): {e}')


def pytest_configure(config):
    # Print installed packages
    if _is_in_ci() and _is_pip_installed():
        print("***** Installed packages *****", flush=True)
        subprocess.check_call([sys.executable, '-m', 'pip', 'freeze', '--all'])
    if config.pluginmanager.hasplugin("xdist"):
        config.pluginmanager.register(DeferPlugin())
    _configure_ascend_dtype_filter(config)


def pytest_report_header(config):
    try:
        return _ascend_dtypes().describe()
    except Exception:  # pragma: no cover
        return None


def pytest_collection_modifyitems(config, items):
    """把 dtype 参数化用例里"NPU 不支持"的组合标记为 skip。

    单个用例/模块可用 ``@pytest.mark.ascend_dtype_filter_off`` 豁免
    (例如过滤策略自身的单元测试需要真的跑 float64)。
    """
    try:
        module = _ascend_dtypes()
        policy = module.policy()
    except Exception:  # pragma: no cover
        return
    if not policy.enabled:
        return
    skipped = 0
    for item in items:
        if item.get_closest_marker('ascend_dtype_filter_off') is not None:
            continue
        hits = [d for d in module.item_dtypes(item) if policy.is_skipped(d)]
        if hits:
            item.add_marker(pytest.mark.skip(reason=policy.skip_reason(hits)))
            skipped += 1
    if skipped:
        print(f'\nascend dtype filter: {skipped} 个 dtype 参数化用例标记为 '
              f'skip ({", ".join(policy.names)})')


# https://docs.pytest.org/en/latest/how-to/writing_hook_functions.html#optionally-using-hooks-from-3rd-party-plugins
def _get_visible_devices_env_var():
    """Return the device visibility env var for the installed backend.

    Ascend NPU: CANN supports `ASCEND_RT_VISIBLE_DEVICES` (same semantics as
    `CUDA_VISIBLE_DEVICES`: comma-separated device IDs, logically renumbered
    from 0), so `cupy.xpu.Device(n)` addresses the n-th visible device.
    """
    try:
        from cupy.backends.backend import is_ascend
        if is_ascend:
            return 'ASCEND_RT_VISIBLE_DEVICES'
    except Exception:
        pass
    return 'CUDA_VISIBLE_DEVICES'


class DeferPlugin:
    """Simple plugin to defer pytest-xdist hook functions."""

    # Edit the device visibility environment variable (`CUDA_VISIBLE_DEVICES`
    # for CUDA, `ASCEND_RT_VISIBLE_DEVICES` for Ascend NPU) for each session.
    # Cannot use `pytest_configure_node` nor `pytest_testnodeready` hook,
    # because they are called in the `master` node (process).
    # See also https://github.com/pytest-dev/pytest-xdist/issues/179.
    @pytest.fixture(autouse=True, scope='session')
    def _rotate_visible_devices(self, worker_id):
        if worker_id == 'master':
            # `worker_id` can be `master` if `pytest-xdist` is installed and
            # run without `-n` option.
            return

        env_var = _get_visible_devices_env_var()

        n_gpu = os.environ.get('CUPY_TEST_GPU_LIMIT')
        if n_gpu is None:
            print('Tip: when using pytest-xdist, you can automatically rotate'
                  f' {env_var} for each test worker by setting'
                  ' CUPY_TEST_GPU_LIMIT environment variable.')
            return
        n_gpu = int(n_gpu)

        assert worker_id.startswith('gw')
        w = int(worker_id[2:])

        devices = os.environ.get(env_var)
        if devices is None:
            devices = [str(k) for k in range(n_gpu)]
        else:
            devices = devices.split(',')[:n_gpu]
        devices = collections.deque(devices)
        devices.rotate(w)
        devices = ','.join(devices)
        os.environ[env_var] = devices
        # With PyTest's default, the print will be shown as
        # "--- Captured stdout setup ---" on failure.
        print(f'{env_var}={devices}')


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
