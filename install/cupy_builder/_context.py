from __future__ import annotations

import dataclasses
import glob
import hashlib
import os
import sys
from typing import TYPE_CHECKING

import cupy_builder

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cupy_builder._features import Feature


def _get_env_bool(
    name: str, env: Mapping[str, str], *, default: bool = False
) -> bool:
    return env[name] != '0' if name in env else default


def _get_env_path(name: str, env: Mapping[str, str]) -> list[str]:
    paths = env.get(name, None)
    if paths is None:
        return []
    return [x for x in paths.split(os.pathsep) if len(x) != 0]


@dataclasses.dataclass
class Context:
    source_root: str
    setup_command: str

    use_cuda_python: bool
    use_hip: bool
    use_stub: bool
    include_dirs: list[str]
    library_dirs: list[str]
    long_description_path: str | None
    wheel_metadata_path: str | None
    profile: bool
    linetrace: bool
    annotate: bool
    no_rpath: bool
    features: dict[str, Feature]
    cupy_cache_key: str
    win32_cl_exe_path: str | None

    # Deprecated
    wheel_libs: list[str]
    wheel_includes: list[str]

    def __init__(
            self, source_root: str, *,
            _env: Mapping[str, str] = os.environ,
            _argv: list[str] = sys.argv) -> None:
        self.source_root = source_root
        self.setup_command = _argv[1] if len(_argv) >= 2 else ''

        self.use_cuda_python = _get_env_bool('CUPY_USE_CUDA_PYTHON', _env)
        self.use_hip = _get_env_bool('CUPY_INSTALL_USE_HIP', _env)

        # Build CuPy with stub header file
        self.use_stub = _get_env_bool("CUPY_INSTALL_USE_STUB", _env)

        # Extra paths to search for libraries/headers during build
        self.include_dirs = _get_env_path('CUPY_INCLUDE_PATH', _env)
        self.library_dirs = _get_env_path('CUPY_LIBRARY_PATH', _env)

        # path to the long description file (reST)
        self.long_description_path = _env.get("CUPY_INSTALL_LONG_DESCRIPTION")
        # wheel metadata (cupy/.data/_wheel.json)
        self.wheel_metadata_path = _env.get("CUPY_INSTALL_WHEEL_METADATA")
        # disable adding default library directories to RPATH
        self.no_rpath = _get_env_bool("CUPY_INSTALL_NO_RPATH", _env)
        # enable profiling for Cython code
        self.profile = _get_env_bool("CUPY_INSTALL_PROFILE", _env)
        # enable coverage for Cython code
        self.annotate = self.linetrace = _get_env_bool(
            "CUPY_INSTALL_COVERAGE", _env)

        if os.environ.get('READTHEDOCS', None) == 'True':
            self.use_stub = True

        self.features = cupy_builder.get_features(self)

        # Calculate cache key for this build
        print('Generating cache key from header files...')
        include_pattern = os.path.join(
            source_root, 'cupy', '_core', 'include', '**')
        include_files = [
            f for f in sorted(glob.glob(include_pattern, recursive=True))
            if os.path.isfile(f)
        ]
        hasher = hashlib.sha1(usedforsecurity=False)
        for include_file in include_files:
            with open(include_file, 'rb') as f:
                relpath = os.path.relpath(include_file, source_root)
                hasher.update(relpath.encode())
                hasher.update(f.read())
                hasher.update(b'\x00')
        cache_key = hasher.hexdigest()
        print(f'Cache key ({len(include_files)} files '
              f'matching {include_pattern}): {cache_key}')
        self.cupy_cache_key = cache_key

        # Host compiler path for Windows, see `_command.py`.
        self.win32_cl_exe_path = None

        # Deprecated
        self.wheel_libs = []
        self.wheel_includes = []
