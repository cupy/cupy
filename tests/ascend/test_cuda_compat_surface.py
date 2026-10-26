"""`cupy.cuda.<X>` 兼容层的回归测试（examples/ 直接依赖它）。

背景：本 fork 把设备 API 搬到了后端无关的 `cupy.xpu`（`cupy/cuda/*.pyx` ->
`cupy/xpu/*.pyx`），于是

  * `cupy/__init__.py` 少了 `from cupy import cuda` —— `import cupy` 不会自动
    导入子模块，`cupy.cuda.Stream` 直接 AttributeError；
  * `cupy/cuda/__init__.py` 只 import 了子模块（`from cupy.xpu import stream`），
    没有重新导出**类/函数** —— `cupy.cuda.Stream` / `cupy.cuda.Device` 同样
    AttributeError。

examples/ 与大量用户代码用的正是这些名字（实测：`cupy.cuda.Stream` 13 处、
`cupy.cuda.Device` 17 处、`cupy.cuda.stream` 22 处、`cupy.cuda.get_elapsed_time`
3 处 ...）。这里不碰设备，只断言名字能解析。
"""

from __future__ import annotations

import pytest

#: examples/ 里实际用到的 `cupy.cuda.*` 名字（grep examples/ 得到）
EXAMPLES_SURFACE = (
    'cuda',
    'cuda.device',
    'cuda.stream',
    'cuda.runtime',
    'cuda.Device',
    'cuda.Stream',
    'cuda.Event',
    'cuda.MemoryPool',
    'cuda.PinnedMemoryPool',
    'cuda.alloc_pinned_memory',
    'cuda.set_pinned_memory_allocator',
    'cuda.set_allocator',
    'cuda.get_elapsed_time',
    'cuda.is_available',
    'cuda.get_local_runtime_version',
)


def _resolve(path):
    import cupy
    obj = cupy
    for part in path.split('.'):
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize('path', EXAMPLES_SURFACE)
def test_cupy_cuda_surface_resolves_after_import_cupy(path):
    """只要 `import cupy`，`cupy.cuda.*` 的这些名字就必须存在（无需 NPU）。"""
    assert _resolve(path) is not None


def test_cuda_and_xpu_expose_the_same_classes():
    """兼容层不能各导出一份（否则 isinstance 判断会挂）。"""
    import cupy
    for name in ('Device', 'Stream', 'Event', 'MemoryPool', 'PinnedMemoryPool'):
        assert getattr(cupy.cuda, name) is getattr(cupy.xpu, name), name


def test_cupy_init_imports_cuda_submodule():
    """`import cupy` 之后 `cupy.cuda` 必然可访问：`cupy/__init__.py` 里要有它。"""
    import os

    import cupy
    src = open(os.path.join(os.path.dirname(cupy.__file__), '__init__.py')).read()
    assert 'from cupy import cuda' in src
    assert hasattr(cupy, 'cuda')
