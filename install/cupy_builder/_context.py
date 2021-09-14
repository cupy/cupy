import os
import typing


def _get_env_bool(name: str, default: bool, env: typing.Mapping) -> bool:
    return env[name] != '0' if name in env else default


class Context:
    def __init__(
            self, source_root: str, *, _env: typing.Mapping = os.environ):
        self.source_root = source_root

        self.enable_thrust = _get_env_bool(
            'CUPY_SETUP_ENABLE_THRUST', True, _env)
        self.use_cuda_python = _get_env_bool(
            'CUPY_USE_CUDA_PYTHON', False, _env)
        self.use_hip = _get_env_bool(
            'CUPY_INSTALL_USE_HIP', False, _env)


def parse_args():
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
    # parser.add_argument(
    #     '--cupy-use-hip', action='store_true', default=False,
    #     help='build CuPy with HIP')

    opts, sys.argv = parser.parse_known_args(sys.argv)

    arg_options = {
        'package_name': opts.cupy_package_name,
        'long_description': opts.cupy_long_description,
        'wheel_libs': opts.cupy_wheel_lib,  # list
        'wheel_includes': opts.cupy_wheel_include,  # list
        'wheel_metadata': opts.cupy_wheel_metadata,
        'no_rpath': opts.cupy_no_rpath,
        'profile': opts.cupy_profile,
        'linetrace': opts.cupy_coverage,
        'annotate': opts.cupy_coverage,
        'no_cuda': opts.cupy_no_cuda,
        'use_hip': use_hip  # opts.cupy_use_hip,
    }
    if check_readthedocs_environment():
        arg_options['no_cuda'] = True
    return arg_options
