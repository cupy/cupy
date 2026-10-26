"""Per-backend build descriptors and the backend registry.

This package answers *"how do I build for backend X?"* in contrast to
``cupy_builder.features`` which answers *"what do I build for backend X?"*.

The registry maps a ``Context`` to the matching :class:`Backend` instance.
To add a new backend, create ``<name>.py`` with a ``Backend`` subclass and
register it in :data:`BACKEND_CLASSES` below. See ``install/README.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cupy_builder.backends._base import Backend
from cupy_builder.backends.ascend import AscendBackend
from cupy_builder.backends.cuda import CudaBackend
from cupy_builder.backends.rocm import RocmBackend

if TYPE_CHECKING:
    from cupy_builder._context import Context

__all__ = [
    'Backend',
    'AscendBackend',
    'CudaBackend',
    'RocmBackend',
    'get_backend',
    'get_backend_by_name',
]


#: Registry of known backends, keyed by :attr:`Backend.name`.
BACKEND_CLASSES: dict[str, type[Backend]] = {
    CudaBackend.name: CudaBackend,
    RocmBackend.name: RocmBackend,
    AscendBackend.name: AscendBackend,
}


# Cache so repeated lookups return the same instance (backends may cache
# version detection results between calls).
_instances: dict[str, Backend] = {}


#: Aliases for backend names that do not have their own descriptor.
#: A "stub" build (RTD / no device) uses CUDA-shaped headers and flags, so it
#: reuses the CUDA backend; its versions are zeroed in compile_time_env.
#: ``ctx.get_backend_name()`` historically returns ``'hip'`` for ROCm, so map
#: it to the ``RocmBackend`` descriptor (whose canonical name is ``'rocm'``).
_BACKEND_ALIASES: dict[str, str] = {
    'stub': CudaBackend.name,
    'hip': RocmBackend.name,
}


def resolve_backend_name(name: str) -> str:
    """Map an alias (e.g. ``'stub'``) to a real backend name."""
    return _BACKEND_ALIASES.get(name, name)


def get_backend_by_name(name: str) -> Backend:
    """Return the singleton :class:`Backend` for ``name``."""
    name = resolve_backend_name(name)
    if name not in BACKEND_CLASSES:
        raise KeyError('Unknown backend: %r' % name)
    if name not in _instances:
        _instances[name] = BACKEND_CLASSES[name]()
    return _instances[name]


def get_backend(ctx: Context) -> Backend:
    """Return the :class:`Backend` matching the build ``ctx``.

    Resolution order mirrors the historical ``ctx.get_backend_name()``:
    stub -> hip -> ascend -> cuda.
    """
    if getattr(ctx, 'use_stub', False):
        return get_backend_by_name('stub')
    if getattr(ctx, 'use_hip', False):
        return get_backend_by_name(RocmBackend.name)
    if getattr(ctx, 'use_ascend', False):
        return get_backend_by_name(AscendBackend.name)
    return get_backend_by_name(CudaBackend.name)
