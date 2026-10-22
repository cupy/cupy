from __future__ import annotations


import dataclasses
import glob
import hashlib
import json
import os
import runpy
import sys
import zlib
from typing import TYPE_CHECKING

import cupy_builder

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


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
    include_dirs: list[str]
    library_dirs: list[str]
    long_description_path: str | None
    wheel_metadata_path: str | None
    profile: bool
    linetrace: bool
    annotate: bool
    no_rpath: bool
    features: dict[str, cupy_builder.Feature]
    cupy_cache_key: str | None
    header_cache: str | None
    win32_cl_exe_path: str | None
    dev_configure_cache: bool
    dev_configure_cache_key: str

    # Deprecated
    wheel_libs: list[str]
    wheel_includes: list[str]

    def __init__(
            self, source_root: str, *,
            _env: Mapping[str, str] = os.environ,
            _argv: Sequence[str] = sys.argv) -> None:
        self.source_root = source_root
        self.setup_command = _argv[1] if len(_argv) >= 2 else ''

        self.use_cuda_python = _get_env_bool('CUPY_USE_CUDA_PYTHON', _env)
        self.use_hip = _get_env_bool('CUPY_INSTALL_USE_HIP', _env)

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
        self.profile = _get_env_bool("CUPY_INSTALL_CYTHON_PROFILE", _env)
        # enable coverage for Cython code
        self.annotate = self.linetrace = _get_env_bool(
            "CUPY_INSTALL_CYTHON_COVERAGE", _env)

        self.features = cupy_builder.get_features(self)

        # Cache keys for this build.
        self.cupy_cache_key = None
        self.header_cache = None

        # Host compiler path for Windows, see `_command.py`.
        self.win32_cl_exe_path = None

        # EXPERIMENTAL: Persist the build configuration to a cache file to
        # skip re-configuring modules when rebuilding during development.
        # Only effective in editable mode (i.e. `pip install -e .`).
        # This is solely intended for use by CuPy developers.
        # End users should NEVER use this flag.
        self.dev_configure_cache = (
            _get_env_bool("CUPY_INSTALL_CONFIGURE_CACHE", _env)
            and self.setup_command == "editable_wheel")
        if self.dev_configure_cache:
            self.calculate_dev_cache_key()

        # Deprecated
        self.wheel_libs = []
        self.wheel_includes = []

    def calculate_cupy_cache_key(self) -> None:
        print('Generating kernel cache keys from header files...')
        include_dir = os.path.join(
            self.source_root, 'cupy', '_core', 'include')
        # Load the shared classifier without importing the CuPy extensions.
        module = runpy.run_path(os.path.join(
            self.source_root, 'cupy', '_core', '_header_cache.py'))
        cache = module['build_header_cache'](include_dir)
        self.cupy_cache_key = cache['keys']['common']
        # Keep the exact bundled-header inventory in the extension. Compress
        # it so the compiler definition stays below command-line size limits.
        self.header_cache = zlib.compress(json.dumps(
            cache, sort_keys=True, separators=(',', ':')).encode()).hex()
        for group, key in cache['keys'].items():
            print(f'Kernel header cache key ({group}): {key}')

    def calculate_dev_cache_key(self) -> None:
        print('Generating configure cache key...')
        pattern = os.path.join(
            self.source_root, 'install', 'cupy_builder', '**')
        cache_key, count = self._calculate_checksum(pattern)
        print(f'Configure cache key ({count} files '
              f'matching {pattern}): {cache_key}')
        self.dev_configure_cache_key = cache_key

    def _calculate_checksum(self, glob_pattern: str) -> tuple[str, int]:
        files = [
            f for f in sorted(glob.glob(glob_pattern, recursive=True))
            if os.path.isfile(f)
        ]
        hasher = hashlib.sha1(usedforsecurity=False)
        for path in files:
            with open(path, 'rb') as f:
                relpath = os.path.relpath(path, self.source_root)
                hasher.update(relpath.encode())
                hasher.update(f.read())
                hasher.update(b'\x00')
        return hasher.hexdigest(), len(files)
