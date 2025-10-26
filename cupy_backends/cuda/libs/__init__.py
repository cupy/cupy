from __future__ import annotations

import importlib as _importlib
import os as _os

from cupy_backends.cuda.api import runtime as _runtime

if not _runtime.is_hip:
    from cuda import pathfinder as _pathfinder


# All submodules are already made lazily loaded (and they should continue
# remaining so). The problem we are solving is we want to load the required
# C library before the submodule that needs it is imported. This list of
# submodules are supported by the pathfinder.
#
# Note: Do NOT add "nvrtc" here. It is run-time linked instead of dynamically
# linked, so its loading is further deferred (to inside SoftLink).
# TODO(leofang): add cutensor (NVIDIA/cuda-python#1144)
# TODO(leofang): add cusparselt
_submodules = (
    'cublas', 'cusolver', 'cusparse', 'curand', 'cufft', 'nccl',
)


def __getattr__(name):
    if not _runtime.is_hip and name in _submodules:
        try:
            _pathfinder.load_nvidia_dynamic_lib(name)
        except _pathfinder.DynamicLibNotFoundError as e:
            if not (_os.environ.get('READTHEDOCS') == 'True'):
                raise ImportError(str(e)) from e
    return _importlib.import_module(f'cupy_backends.cuda.libs.{name}')
