import argparse
import os
import sys
from typing import Any, List, Mapping, Optional, Tuple

from cupy_builder import install_utils


def _get_env_bool(name: str, default: bool, env: Mapping[str, str]) -> bool:
    return env[name] != '0' if name in env else default


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

        if os.environ.get('READTHEDOCS', None) == 'True':
            self.use_stub = True

        self._generate_translation_units()
        print(f"\n\n\n{self.module_TUs}\n\n\n")

    def __del__(self):
        # TODO(leofang): keep them if generating sdist or in debug mode?
        if True:
            for mod, files in self.module_TUs.items():
                for f in files:
                    os.remove(f)

    def _generate_translation_units(self):
        # TODO(leofang): move these data to _modules.py
        data = {
            'thrust': {
                'argsort': f'{self.source_root}/cupy/cuda/cupy_thrust_argsort.template',
                'lexsort': f'{self.source_root}/cupy/cuda/cupy_thrust_lexsort.template',
                'sort': f'{self.source_root}/cupy/cuda/cupy_thrust_sort.template',
            },
            'cub': {
            }
        }
        # TODO(leofang): some functions only support a subset of this list
        type_to_code = {
            'char': 'CUPY_TYPE_INT8',
            'short': 'CUPY_TYPE_INT16',
            'int': 'CUPY_TYPE_INT32',
            'int64_t': 'CUPY_TYPE_INT64',
            'unsigned char': 'CUPY_TYPE_UINT8',
            'unsigned short': 'CUPY_TYPE_UINT16',
            'unsigned int': 'CUPY_TYPE_UINT32',
            'uint64_t': 'CUPY_TYPE_UINT64',
            '__half': 'CUPY_TYPE_FLOAT16',
            'float': 'CUPY_TYPE_FLOAT32',
            'double': 'CUPY_TYPE_FLOAT64',
            'complex<float>': 'CUPY_TYPE_COMPLEX64',
            'complex<double>': 'CUPY_TYPE_COMPLEX128',
            'bool': 'CUPY_TYPE_BOOL',
        }

        module_TUs = {}
        for mod, funcs in data.items():
            TUs = []
            for func_name, template_path in funcs.items():
                for type_name, code_name in type_to_code.items():
                    TUs.append(install_utils.generate_translation_unit(
                        func_name, type_name, code_name, template_path)
                    )
            module_TUs[mod] = TUs

        self.module_TUs = module_TUs


def parse_args(argv: List[str]) -> Tuple[Any, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        '--cupy-package-name', type=str, default='cupy',
        help='alternate package name')
    parser.add_argument(
        '--cupy-long-description', type=str, default=None,
        help='path to the long description file')
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
