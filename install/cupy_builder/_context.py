import argparse
import glob
import hashlib
import os
import sys
from typing import Any, List, Mapping, Optional, Tuple

import cupy_builder


def _get_env_bool(name: str, default: bool, env: Mapping[str, str]) -> bool:
    return env[name] != '0' if name in env else default


def _get_env_path(name: str, env: Mapping[str, str]) -> List[str]:
    paths = env.get(name, None)
    if paths is None:
        return []
    return [x for x in paths.split(os.pathsep) if len(x) != 0]


class Context:
    def __init__(
            self, source_root: str, *,
            _env: Mapping[str, str] = os.environ,
            _argv: List[str] = sys.argv):
        self.source_root = source_root

        self.use_cuda_python = _get_env_bool(
            'CUPY_USE_CUDA_PYTHON', False, _env)
        self.use_hip = _get_env_bool(
            'CUPY_INSTALL_USE_HIP', False, _env)
        self.include_dirs = _get_env_path('CUPY_INCLUDE_PATH', _env)
        self.library_dirs = _get_env_path('CUPY_LIBRARY_PATH', _env)

        cmdopts, _argv[:] = parse_args(_argv)
        self.package_name: str = cmdopts.cupy_package_name
        self.long_description_path: Optional[str] = (
            cmdopts.cupy_long_description)
        self.wheel_libs: List[str] = cmdopts.cupy_wheel_lib
        self.wheel_includes: List[str] = cmdopts.cupy_wheel_include
        self.wheel_metadata_path: Optional[str] = (
            cmdopts.cupy_wheel_metadata)
        self.no_rpath: bool = cmdopts.cupy_no_rpath
        self.profile: bool = cmdopts.cupy_profile
        self.linetrace: bool = cmdopts.cupy_coverage
        self.annotate: bool = cmdopts.cupy_coverage
        self.use_stub: bool = cmdopts.cupy_no_cuda

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
                hasher.update(include_file.encode())
                hasher.update(f.read())
                hasher.update(b'\x00')
        cache_key = hasher.hexdigest()
        print(f'Cache key ({len(include_files)} files '
              f'matching {include_pattern}): {cache_key}')
        self.cupy_cache_key = cache_key

        # Host compiler path for Windows, see `_command.py`.
        self.win32_cl_exe_path: Optional[str] = None


def parse_args(argv: List[str]) -> Tuple[Any, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        '--cupy-package-name', type=str, default='cupy',
        help='alternate package name')
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
