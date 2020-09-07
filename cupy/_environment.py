"""
This file must not depend on any other CuPy modules.
"""

import ctypes
import json
import os
import os.path
import shutil
import sys
import warnings


# '' for uninitialized, None for non-existing
_cuda_path = ''
_nvcc_path = ''
_cub_path = ''

"""
Library Preloading
------------------

Wheel packages are built against specific versions of CUDA libraries
(cuTENSOR/NCCL/cuDNN).
To avoid loading wrong version, these shared libraries are manually
preloaded.

# TODO(kmaehashi) Currently cuDNN only. Support cuTENSOR and NCCL.

Example of `_preload_config` is as follows:

{
    # CUDA version string
    'cuda': '11.0',

    'cudnn': {
        # cuDNN version string
        'version': '8.0.0',

        # name of the shared library
        'filename': 'libcudnn.so.X.Y.Z'  # or `cudnn64_X.dll` for Windows
    }
}

The configuration file is intended solely for internal purposes and
not expected to be parsed by end-users.
"""

_preload_config = None

_preload_libs = {
    'cudnn': None,
    # 'nccl': None,
    # 'cutensor': None,
}

_preload_logs = []


def _log(msg):
    # TODO(kmaehashi): replace with the standard logging
    _preload_logs.append(msg)


def get_cuda_path():
    # Returns the CUDA installation path or None if not found.
    global _cuda_path
    if _cuda_path == '':
        _cuda_path = _get_cuda_path()
    return _cuda_path


def get_nvcc_path():
    # Returns the path to the nvcc command or None if not found.
    global _nvcc_path
    if _nvcc_path == '':
        _nvcc_path = _get_nvcc_path()
    return _nvcc_path


def get_cub_path():
    # Returns the CUB header path or None if not found.
    global _cub_path
    if _cub_path == '':
        _cub_path = _get_cub_path()
    return _cub_path


def _get_cuda_path():
    # Use environment variable
    cuda_path = os.environ.get('CUDA_PATH', '')  # Nvidia default on Windows
    if os.path.exists(cuda_path):
        return cuda_path

    # Use nvcc path
    nvcc_path = shutil.which('nvcc')
    if nvcc_path is not None:
        return os.path.dirname(os.path.dirname(nvcc_path))

    # Use typical path
    if os.path.exists('/usr/local/cuda'):
        return '/usr/local/cuda'

    return None


def _get_nvcc_path():
    # Honor the "NVCC" env var
    nvcc_path = os.environ.get('NVCC', None)
    if nvcc_path is not None:
        return nvcc_path

    # Lookup <CUDA>/bin
    cuda_path = get_cuda_path()
    if cuda_path is None:
        return None

    return shutil.which('nvcc', path=os.path.join(cuda_path, 'bin'))


def _get_cub_path():
    # runtime discovery of CUB headers
    cuda_path = get_cuda_path()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if os.path.isdir(os.path.join(current_dir, 'core/include/cupy/cub')):
        _cub_path = '<bundle>'
    elif cuda_path is not None and os.path.isdir(
            os.path.join(cuda_path, 'include/cub')):
        # use built-in CUB for CUDA 11+
        _cub_path = '<CUDA>'
    else:
        _cub_path = None
    return _cub_path


def _setup_win32_dll_directory():
    # Setup DLL directory to load CUDA Toolkit libs and shared libraries
    # added during the build process.
    if sys.platform.startswith('win32'):
        # Path to the CUDA Toolkit binaries
        cuda_path = get_cuda_path()
        if cuda_path is not None:
            cuda_bin_path = os.path.join(cuda_path, 'bin')
        else:
            cuda_bin_path = None
            warnings.warn(
                'CUDA path could not be detected.'
                ' Set CUDA_PATH environment variable if CuPy fails to load.')
        _log('CUDA_PATH: {}'.format(cuda_path))

        # Path to shared libraries in wheel
        wheel_libdir = os.path.join(
            get_cupy_install_path(), 'cupy', '.data', 'lib')
        if os.path.isdir(wheel_libdir):
            _log('Wheel shared libraries: {}'.format(wheel_libdir))
        else:
            _log('Not wheel distribution ({} not found)'.format(
                wheel_libdir))
            wheel_libdir = None

        if (3, 8) <= sys.version_info:
            if cuda_bin_path is not None:
                _log('Adding DLL search path: {}'.format(cuda_bin_path))
                os.add_dll_directory(cuda_bin_path)
            if wheel_libdir is not None:
                _log('Adding DLL search path: {}'.format(wheel_libdir))
                os.add_dll_directory(wheel_libdir)
        else:
            # Users are responsible for adding `%CUDA_PATH%/bin` to PATH.
            if wheel_libdir is not None:
                _log('Adding to PATH: {}'.format(wheel_libdir))
                path = os.environ.get('PATH', '')
                os.environ['PATH'] = wheel_libdir + os.pathsep + path


def get_cupy_install_path():
    # Path to the directory where the package is installed.
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))


def get_cupy_cuda_lib_path():
    """Returns the directory where CUDA external libraries are installed.

    This environment variable only affects wheel installations.

    Shared libraries are looked up from:
    `$CUPY_CUDA_LIB_PATH/$CUDA_VERSION/$LIBRARY_NAME/$LIBRARY_VERSION/lib64`
    (`bin` instead of `lib64` on Windows)

    The default path is `~/.cupy/cuda_lib`.
    """
    cupy_cuda_lib_path = os.environ.get('CUPY_CUDA_LIB_PATH', None)
    if cupy_cuda_lib_path is None:
        return os.path.expanduser('~/.cupy/cuda_lib')
    return os.path.abspath(cupy_cuda_lib_path)


def get_preload_config():
    global _preload_config
    if _preload_config is None:
        config_path = os.path.join(
            get_cupy_install_path(), 'cupy', '.data', '_wheel.json')
        if not os.path.exists(config_path):
            return None
        _preload_config = json.load(open(config_path))
    return _preload_config


def _preload_libraries():
    """Preload dependent shared libraries.

    The preload configuration file (cupy/.data/_wheel.json) will be added
    during the wheel build process.
    """

    config = get_preload_config()
    if config is None:
        _log('Skip preloading as this is not a wheel installation')
        return

    cuda_version = config['cuda']
    _log('CuPy wheel package built for CUDA {}'.format(cuda_version))

    cupy_cuda_lib_path = get_cupy_cuda_lib_path()
    _log('CuPy CUDA library directory: {}'.format(cupy_cuda_lib_path))

    for lib in _preload_libs.keys():
        if lib not in config:
            _log('Not preloading {}'.format(lib))
            continue
        version = config[lib]['version']
        filename = config[lib]['filename']
        _log('Looking for {} version {} ({})'.format(lib, version, filename))

        lib64dir = 'bin' if sys.platform.startswith('win32') else 'lib64'
        libpath = os.path.join(
            cupy_cuda_lib_path, config['cuda'], lib, version, lib64dir,
            filename)
        if os.path.exists(libpath):
            _log('Trying to load {}'.format(libpath))
            try:
                # Keep reference to the preloaded module.
                _preload_libs[lib] = (libpath, ctypes.CDLL(libpath))
                _log('Loaded')
            except Exception as e:
                msg = 'CuPy failed to preload library ({}): {} ({})'.format(
                    libpath, type(e).__name__, str(e))
                _log(msg)
                warnings.warn(msg)
        else:
            _log('File {} could not be found'.format(libpath))

            # Lookup library with fully-qualified version (e.g.,
            # `libcudnn.so.X.Y.Z`).
            _log('Trying to load {}'.format(filename))
            try:
                _preload_libs[lib] = (filename, ctypes.CDLL(filename))
                _log('Loaded')
            except Exception as e:
                # Fallback to the standard shared library lookup which only
                # uses the major version (e.g., `libcudnn.so.X`).
                _log('Library {} could not be preloaded: {}'.format(lib, e))


def _get_preload_logs():
    return '\n'.join(_preload_logs)


def _preload_warning(lib, exc):
    config = get_preload_config()
    if config is not None and lib in config:
        warnings.warn('''
{lib} library could not be loaded.

Reason: {exc_type} ({exc})

You can install the library by:
  $ python -m cupyx.tools.install_library --library {lib} --cuda {cuda}
'''.format(lib=lib, exc_type=type(exc).__name__, exc=str(exc),
           cuda=config['cuda']))
