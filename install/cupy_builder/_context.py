from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import os
import sys
from typing import TYPE_CHECKING, Any

import cupy_builder

if TYPE_CHECKING:
    from collections.abc import Mapping

    from install.cupy_builder._features import Feature


def _get_env_bool(name: str, default: bool, env: Mapping[str, str]) -> bool:
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
    wheel_libs: list[str]
    wheel_includes: list[str]
    wheel_metadata_path: str | None
    no_rpath: bool
    profile: bool
    linetrace: bool
    annotate: bool
    use_stub: bool
    no_rpath: bool
    use_stub: bool
    features: dict[str, Feature]
    cupy_cache_key: str
    win32_cl_exe_path: str | None

    def __init__(
            self, source_root: str, *,
            _env: Mapping[str, str] = os.environ,
            _argv: list[str] = sys.argv):
        self.source_root = source_root
        self.setup_command = _argv[1] if len(_argv) >= 2 else ''

        self.use_cuda_python = _get_env_bool(
            'CUPY_USE_CUDA_PYTHON', False, _env)
        self.use_hip = _get_env_bool(
            'CUPY_INSTALL_USE_HIP', False, _env)
        self.include_dirs = _get_env_path('CUPY_INCLUDE_PATH', _env)
        self.library_dirs = _get_env_path('CUPY_LIBRARY_PATH', _env)

        cmdopts, _argv[:] = parse_args(_argv)
        self.long_description_path = (
            cmdopts.cupy_long_description)
        self.wheel_libs = cmdopts.cupy_wheel_lib
        self.wheel_includes = cmdopts.cupy_wheel_include
        self.wheel_metadata_path = (
            cmdopts.cupy_wheel_metadata)
        self.no_rpath = cmdopts.cupy_no_rpath
        self.profile = cmdopts.cupy_profile
        self.linetrace = cmdopts.cupy_coverage
        self.annotate = cmdopts.cupy_coverage
        self.use_stub = cmdopts.cupy_no_cuda

        if _get_env_bool('CUPY_INSTALL_NO_RPATH', False, _env):
            self.no_rpath = True

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


def parse_args(argv: list[str]) -> tuple[Any, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        '--cupy-long-description', type=str, default=None,
        help='path to the long description file (reST)')
    parser.add_argument(
        '--cupy-wheel-lib', type=str, action='append', default=[],
        help='shared library to copy into the wheel '
             '(can be specified for multiple times)')
    parser.add_argument(
        '--cupy-wheel-include', type=str, action='append', default=[],
        help='An include file to copy into the wheel. '
             'Delimited by a colon. '
             'The former part is a full path of the source include file and '
             'the latter is the relative path within cupy wheel. '
             '(can be specified for multiple times)')
    parser.add_argument(
        '--cupy-wheel-metadata', type=str, default=None,
        help='wheel metadata (cupy/.data/_wheel.json)')
    parser.add_argument(
        '--cupy-no-rpath', action='store_true', default=False,
        help='disable adding default library directories to RPATH')
    parser.add_argument(
        '--cupy-profile', action='store_true', default=False,
        help='enable profiling for Cython code')
    parser.add_argument(
        '--cupy-coverage', action='store_true', default=False,
        help='enable coverage for Cython code')
    parser.add_argument(
        '--cupy-no-cuda', action='store_true', default=False,
        help='build CuPy with stub header file')

    return parser.parse_known_args(argv)
