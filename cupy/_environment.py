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
_rocm_path = ''
_hipcc_path = ''
_cub_path = ''

"""
Library Preloading
------------------

Wheel packages are built against specific versions of CUDA libraries
(cuTENSOR/NCCL/cuDNN).
To avoid loading wrong version, these shared libraries are manually
preloaded.

# TODO(kmaehashi): Support NCCL

Example of `_preload_config` is as follows:

{
    # installation source
    'packaging': 'pip',

    # CUDA version string
    'cuda': '11.0',

    'cudnn': {
        # cuDNN version string
        'version': '8.0.0',

        # names of the shared library
        'filenames': ['libcudnn.so.X.Y.Z']  # or `cudnn64_X.dll` for Windows
    }
}

The configuration file is intended solely for internal purposes and
not expected to be parsed by end-users.
"""

_preload_config = None

_preload_libs = {
    'cudnn': {},
    'nccl': {},
    'cutensor': {},
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


def get_rocm_path():
    # Returns the ROCm installation path or None if not found.
    global _rocm_path
    if _rocm_path == '':
        _rocm_path = _get_rocm_path()
    return _rocm_path


def get_hipcc_path():
    # Returns the path to the hipcc command or None if not found.
    global _hipcc_path
    if _hipcc_path == '':
        _hipcc_path = _get_hipcc_path()
    return _hipcc_path


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


def _get_rocm_path():
    # Use environment variable
    rocm_path = os.environ.get('ROCM_HOME', '')
    if os.path.exists(rocm_path):
        return rocm_path

    # Use hipcc path
    hipcc_path = shutil.which('hipcc')
    if hipcc_path is not None:
        return os.path.dirname(os.path.dirname(hipcc_path))

    # Use typical path
    if os.path.exists('/opt/rocm'):
        return '/opt/rocm'

    return None


def _get_hipcc_path():
    # TODO(leofang): Introduce an env var HIPCC?

    # Lookup <ROCM>/bin
    rocm_path = get_rocm_path()
    if rocm_path is None:
        return None

    return shutil.which('hipcc', path=os.path.join(rocm_path, 'bin'))


def _get_cub_path():
    # runtime discovery of CUB headers
    from cupy_backends.cuda.api import runtime
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if not runtime.is_hip:
        cuda_path = get_cuda_path()
        if os.path.isdir(os.path.join(current_dir, '_core/include/cupy/cub')):
            _cub_path = '<bundle>'
        elif cuda_path is not None and os.path.isdir(
                os.path.join(cuda_path, 'include/cub')):
            # use built-in CUB for CUDA 11+
            _cub_path = '<CUDA>'
        else:
            _cub_path = None
    else:
        # the bundled CUB does not work in ROCm
        rocm_path = get_rocm_path()
        if rocm_path is not None and os.path.isdir(
                os.path.join(rocm_path, 'include/hipcub')):
            # use hipCUB
            _cub_path = '<ROCm>'
        else:
            _cub_path = None
    return _cub_path


def _setup_win32_dll_directory():
    # Setup DLL directory to load CUDA Toolkit libs and shared libraries
    # added during the build process.
    if sys.platform.startswith('win32'):
        is_conda = ((os.environ.get('CONDA_PREFIX') is not None)
                    or (os.environ.get('CONDA_BUILD_STATE') is not None))
        # Path to the CUDA Toolkit binaries
        cuda_path = get_cuda_path()
        if cuda_path is not None:
            if is_conda:
                cuda_bin_path = cuda_path
            else:
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

    Shared libraries are looked up from
    `$CUPY_CUDA_LIB_PATH/$CUDA_VER/$LIB_NAME/$LIB_VER/{lib,lib64,bin}`,
    e.g., `~/.cupy/cuda_lib/11.2/cudnn/8.1.1/lib64/libcudnn.so.8.1.1`.

    The default $CUPY_CUDA_LIB_PATH is `~/.cupy/cuda_lib`.
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
        with open(config_path) as f:
            _preload_config = json.load(f)
    return _preload_config


def _preload_libraries():
    """Preload dependent shared libraries.

    The preload configuration file (cupy/.data/_wheel.json) will be added
    during the wheel build process.
    """

    config = get_preload_config()
    if (config is None) or (config['packaging'] == 'conda'):
        # We don't do preload if CuPy is installed from Conda-Forge, as we
        # cannot guarantee the version pinned in _wheel.json, which is
        # encoded in config[lib]['filenames'], is always available on
        # Conda-Forge. See here for the configuration files used in
        # Conda-Forge distributions.
        # https://github.com/conda-forge/cupy-feedstock/blob/master/recipe/preload_config/
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
        filenames = config[lib]['filenames']
        for filename in filenames:
            _log(f'Looking for {lib} version {version} ({filename})')

            # "lib": cuTENSOR (Linux/Windows) / NCCL (Linux)
            # "lib64": cuDNN (Linux)
            # "bin": cuDNN (Windows)
            libpath_cands = [
                os.path.join(
                    cupy_cuda_lib_path, config['cuda'], lib, version, x,
                    filename)
                for x in ['lib', 'lib64', 'bin']]
            for libpath in libpath_cands:
                if not os.path.exists(libpath):
                    _log('Rejected candidate (not found): {}'.format(libpath))
                    continue

                try:
                    _log(f'Trying to load {libpath}')
                    # Keep reference to the preloaded module.
                    _preload_libs[lib][libpath] = ctypes.CDLL(libpath)
                    _log('Loaded')
                    break
                except Exception as e:
                    e_type = type(e).__name__  # NOQA
                    msg = (
                        f'CuPy failed to preload library ({libpath}): '
                        f'{e_type} ({e})')
                    _log(msg)
                    warnings.warn(msg)
            else:
                _log('File {} could not be found'.format(filename))

                # Lookup library with fully-qualified version (e.g.,
                # `libcudnn.so.X.Y.Z`).
                _log(f'Trying to load {filename} from default search path')
                try:
                    _preload_libs[lib][filename] = ctypes.CDLL(filename)
                    _log('Loaded')
                except Exception as e:
                    # Fallback to the standard shared library lookup which only
                    # uses the major version (e.g., `libcudnn.so.X`).
                    _log(f'Library {lib} could not be preloaded: {e}')


def _get_preload_logs():
    return '\n'.join(_preload_logs)


def _preload_warning(lib, exc):
    config = get_preload_config()
    if config is not None and lib in config:
        msg = '''
{lib} library could not be loaded.

Reason: {exc_type} ({exc})

You can install the library by:
'''
        if config['packaging'] == 'pip':
            msg += '''
  $ python -m cupyx.tools.install_library --library {lib} --cuda {cuda}
'''
        elif config['packaging'] == 'conda':
            msg += '''
  $ conda install -c conda-forge {lib}
'''
        else:
            raise AssertionError
        msg = msg.format(
            lib=lib, exc_type=type(exc).__name__, exc=str(exc),
            cuda=config['cuda'])
        warnings.warn(msg)


def _detect_duplicate_installation():
    # importlib.metadata only available in Python 3.8+.
    if sys.version_info < (3, 8):
        return
    import importlib.metadata

    # List of all CuPy packages, including out-dated ones.
    known = [
        'cupy',
        'cupy-cuda80',
        'cupy-cuda90',
        'cupy-cuda91',
        'cupy-cuda92',
        'cupy-cuda100',
        'cupy-cuda101',
        'cupy-cuda102',
        'cupy-cuda110',
        'cupy-cuda111',
        'cupy-cuda112',
        'cupy-cuda113',
        'cupy-cuda114',
        'cupy-rocm-4-0',
        'cupy-rocm-4-1',
        'cupy-rocm-4-2',
        'cupy-rocm-4-3',
    ]
    cupy_installed = [
        name for name in known
        if list(importlib.metadata.distributions(name=name))]
    if 1 < len(cupy_installed):
        cupy_packages_list = ', '.join(sorted(cupy_installed))
        warnings.warn(f'''
--------------------------------------------------------------------------------

  CuPy may not function correctly because multiple CuPy packages are installed
  in your environment:

    {cupy_packages_list}

  Follow these steps to resolve this issue:

    1. For all packages listed above, run the following command to remove all
       existing CuPy installations:

         $ pip uninstall <package_name>

      If you previously installed CuPy via conda, also run the following:

         $ conda uninstall cupy

    2. Install the appropriate CuPy package.
       Refer to the Installation Guide for detailed instructions.

         https://docs.cupy.dev/en/stable/install.html

--------------------------------------------------------------------------------
''')
