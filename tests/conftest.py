from __future__ import annotations

import collections
import os
import subprocess
import sys
import random

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

    _configure_ascend_dtype_filter(config)

    if config.pluginmanager.hasplugin("xdist"):
        n_gpu = os.environ.get('CUPY_TEST_GPU_LIMIT')
        worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'master')
        # `CUDA_VISIBLE_DEVICES` for CUDA, `ASCEND_RT_VISIBLE_DEVICES` for NPU
        env_var = _get_visible_devices_env_var()

        if n_gpu is None:
            if worker_id == 'master':
                print('\nTIP: when using pytest-xdist, you can automatically '
                      f'rotate {env_var} for each test worker by setting '
                      'CUPY_TEST_GPU_LIMIT environment variable.\n')
        else:
            # We'll edit the device visibility environment variable for each
            # worker which needs to happen before the device init.
            n_gpu = int(n_gpu)

            devices = os.environ.get(env_var)
            if devices is None:
                devices = [str(k) for k in range(n_gpu)]
            else:
                devices = devices.split(',')[:n_gpu]

            if worker_id == 'master':
                print(f'\nNOTE: Setting workers to use a shifted version of:'
                      f'\n    {env_var}={",".join(devices)}\n')
            else:
                assert worker_id.startswith('gw')
                w = int(worker_id[2:])

                devices = collections.deque(devices)
                # left rotate so second worker gets second GPU
                devices.rotate(-w)
                devices = ','.join(devices)
                os.environ[env_var] = devices


def pytest_report_header(config):
    try:
        return _ascend_dtypes().describe()
    except Exception:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# CUDA-only test modules
#
# Upstream test modules that import CUDA-only cupy internals at module scope
# (nvrtc, cufft, cublas, curand, cusparse, `cupy._core._cub_reduction`,
# `cupy._core._routines_math._scan_for_test`). Those modules are not compiled
# on the Ascend backend (see `install/cupy_builder/features/ascend.py`), so the
# import raises during *collection* and aborts the entire run before a single
# test executes. Skip them at collection time instead.
#
# NB: `test_search.py` / `test_sumprod.py` also carry backend-neutral cases
# (searchsorted, sum/prod); they are listed here because their module-level
# `from cupy._core import _cub_reduction` cannot be satisfied without cub.
_ASCEND_CUDA_ONLY_PATHS = frozenset({
    'cupy_tests/test_cublas.py',
    'cupy_tests/core_tests/test_core.py',
    'cupy_tests/core_tests/test_cub_reduction.py',
    'cupy_tests/core_tests/test_function.py',
    'cupy_tests/core_tests/test_ndarray_reduction.py',
    'cupy_tests/core_tests/test_raw.py',
    'cupy_tests/core_tests/test_scan.py',
    'cupy_tests/fft_tests/test_cache.py',
    'cupy_tests/fft_tests/test_callback.py',
    'cupy_tests/math_tests/test_sumprod.py',
    'cupy_tests/sorting_tests/test_search.py',
})

#: Directories that target the CUDA backend exclusively.
_ASCEND_CUDA_ONLY_DIRS = ('cupy_tests/cuda_tests',)


def _is_ascend_backend():
    try:
        from cupy.backends.backend import is_ascend
        return bool(is_ascend)
    except Exception:
        return False


def pytest_ignore_collect(collection_path, config, **kwargs):
    """Do not collect CUDA-only test modules when running on Ascend NPU."""
    if not _is_ascend_backend():
        return False
    path = collection_path
    if path is None:  # pytest < 8 named this argument `path`
        path = kwargs.get('path')
    if path is None:
        return False
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    rel = os.path.relpath(str(path), tests_dir).replace(os.sep, '/')
    if rel in _ASCEND_CUDA_ONLY_PATHS:
        return True
    return any(rel == d or rel.startswith(d + '/')
               for d in _ASCEND_CUDA_ONLY_DIRS)


#: Ascend 上无条件跳过的 dtype。
#:
#: complex128 在部分 CANN 版本 (如 8.2) 上 aclnn 复数算子缺失/直接报错，
#: 逐个用例失败没有意义，collection 阶段整批跳过。
#: 单个用例/模块可用 ``ascend_dtype_filter_off`` 标记豁免。
_ASCEND_SKIP_ALWAYS_DTYPES = frozenset({numpy.dtype('complex128')})


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
    skipped = 0
    for item in items:
        if item.get_closest_marker('ascend_dtype_filter_off') is not None:
            continue
        try:
            dtypes = module.item_dtypes(item)
        except Exception:  # pragma: no cover
            dtypes = ()
        always = sorted(d for d in dtypes if d in _ASCEND_SKIP_ALWAYS_DTYPES)
        if always:
            item.add_marker(pytest.mark.skip(
                reason='{} not supported by this CANN build'.format(
                    ', '.join(str(d) for d in always))))
            skipped += 1
            continue
        if not policy.enabled:
            continue
        hits = [d for d in dtypes if policy.is_skipped(d)]
        if hits:
            item.add_marker(pytest.mark.skip(reason=policy.skip_reason(hits)))
            skipped += 1
    if skipped:
        print(f'\nascend dtype filter: {skipped} 个 dtype 参数化用例标记为 '
              f'skip ({"complex128, " if _ASCEND_SKIP_ALWAYS_DTYPES else ""}'
              f'{", ".join(policy.names) if policy.enabled else "policy off"})')


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


if int(os.environ.get('CUPY_CI_ENABLE_GCP_KERNEL_CACHE', 0)) != 0:
    from cupy.cuda.compiler import _set_kernel_cache_backend
    from cupy_tests._gcp_kernel_cache_backend import GCPStorageCacheBackend

    backend = GCPStorageCacheBackend(
        bucket_name='tmp-asia-pfn-public-ci',
        prefix='cupy-ci/kernel_cache_objects/'
    )
    print("GCP kernel cache: initializing local cache from GCS...", flush=True)
    backend.initialize_local_cache()
    _set_kernel_cache_backend(backend)


if int(os.environ.get('CUPY_TEST_RANDOM_SUBSAMPLE', '0')):
    seed = int(os.environ.get('_CUPY_TEST_SUBSAMPLE_SEED', '-1'))
    if seed == -1:
        # Generate seed and store as environment variable for pytest-xdist
        # (and maybe to be picked up the `for_all_dtypes`).
        import secrets

        seed = secrets.randbits(64)
        os.environ['_CUPY_TEST_SUBSAMPLE_SEED'] = str(seed)

    rng = random.Random(seed)

    def pytest_collection_modifyitems(session, config, items):
        original_len = len(items)
        if original_len == 0:
            return  # No tests, so nothing to subsample

        groups = collections.defaultdict(list)
        for item in items:
            base = f"{item.parent.nodeid}::{item.originalname}"
            groups[base].append(item)
        items[:] = [
            x for v in groups.values()
            for x in rng.sample(v, 1 + len(v) // 15)
        ]

        print(f"\nRandomly subsampled parametrized tests from {original_len} "
              f"to {len(items)} ({len(items) / original_len:.1%}). "
              f"(Similar subsampling is also done for `for_all_dtypes`!)\n")

        # Disable strict xfail on selected items — subsampling (e.g. via
        # for_all_dtypes) may remove the failing parametrizations.
        # (This also overrides the global pyproject setting.)
        # TODO(seberg): Could audit strict xfail to remove this.
        for item in items:
            for node in item.listchain():
                for i, marker in enumerate(node.own_markers):
                    if marker.name == 'xfail':
                        kwargs = dict(marker.kwargs)
                        kwargs['strict'] = False
                        node.own_markers[i] = pytest.mark.xfail(
                            *marker.args, **kwargs).mark
