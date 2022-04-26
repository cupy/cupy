import ctypes
import pkg_resources
import os
import sys
from typing import List, Optional

from setuptools import setup


VERSION = '10.4.0'

# List of packages supported by this version of CuPy.
PACKAGES = [
    'cupy-cuda102',
    'cupy-cuda110',
    'cupy-cuda111',
    'cupy-cuda112',
    'cupy-cuda113',
    'cupy-cuda114',
    'cupy-cuda115',
    'cupy-cuda116',
    'cupy-rocm-4-0',
    'cupy-rocm-4-2',
    'cupy-rocm-4-3',
    'cupy-rocm-5-0',
]

# List of packages NOT supported by this version of CuPy.
PACKAGES_OUTDATED = [
    'cupy-cuda80',
    'cupy-cuda90',
    'cupy-cuda91',
    'cupy-cuda92',
    'cupy-cuda100',
    'cupy-cuda101',
]

# List of sdist packages.
PACKAGES_SDIST = [
    'cupy',
]


class AutoDetectionFailed(Exception):
    pass


def _log(msg: str) -> None:
    sys.stdout.write(f'[cupy-wheel] {msg}\n')
    sys.stdout.flush()


def _get_version_from_library(
        libnames: List[str], funcname: str) -> Optional[int]:
    """Returns the library version from list of candidate libraries."""

    for libname in libnames:
        try:
            _log(f'Looking for library: {libname}')
            runtime_so = ctypes.CDLL(libname)
            break
        except Exception as e:
            _log(f'Failed to open {libname}: {e}')
    else:
        _log('No more candidate library to find')
        return None

    func = getattr(runtime_so, funcname, None)
    if func is None:
        raise AutoDetectionFailed(
            f'{libname}: {func} could not be found')
    func.restype = ctypes.c_int  # cudaError_t
    func.argtypes = [ctypes.POINTER(ctypes.c_int)]
    version_ptr = ctypes.c_int()
    retval = func(version_ptr)
    if retval != 0:
        raise AutoDetectionFailed(
            f'{libname}: {func} returned error: {retval}')
    version = version_ptr.value
    _log(f'Detected version: {version}')
    return version


def _get_cuda_version() -> Optional[int]:
    """Returns the detected CUDA version or None."""

    if sys.platform == 'linux':
        libnames = ['libcudart.so']
    elif sys.platform == 'win32':
        libnames = ['cudart64_110.dll', 'cudart64_102.dll']
    else:
        _log(f'CUDA detection unsupported on platform: {sys.platform}')
        return None
    _log(f'Trying to detect CUDA version from libraries: {libnames}')
    version = _get_version_from_library(libnames, 'cudaRuntimeGetVersion')
    return version


def _get_rocm_version() -> Optional[int]:
    """Returns the detected ROCm version or None."""
    if sys.platform == 'linux':
        libnames = ['libamdhip64.so']
    else:
        _log(f'ROCm detection unsupported on platform: {sys.platform}')
        return None
    version = _get_version_from_library(libnames, 'hipRuntimeGetVersion')
    return version


def _find_installed_packages() -> List[str]:
    """Returns the list of CuPy packages installed in the environment."""

    found = []
    for pkg in (PACKAGES + PACKAGES_OUTDATED + PACKAGES_SDIST):
        try:
            pkg_resources.get_distribution(pkg)
            found.append(pkg)
        except pkg_resources.DistributionNotFound:
            pass
    return found


def _cuda_version_to_package(ver: int) -> str:
    if ver < 10020:
        raise AutoDetectionFailed(
            f'Your CUDA version ({ver}) is too old.')
    elif ver < 11000:
        # CUDA 10.2
        suffix = '102'
    elif ver < 11010:
        # CUDA 11.0
        suffix = '110'
    elif ver < 11020:
        # CUDA 11.1
        suffix = '111'
    elif ver < 11030:
        # CUDA 11.2
        suffix = '112'
    elif ver < 11040:
        # CUDA 11.3
        suffix = '113'
    elif ver < 11050:
        # CUDA 11.4
        suffix = '114'
    elif ver < 11060:
        # CUDA 11.5
        suffix = '115'
    elif ver < 11070:
        # CUDA 11.6
        suffix = '116'
    else:
        raise AutoDetectionFailed(
            f'Your CUDA version ({ver}) is too new.')
    return f'cupy-cuda{suffix}'


def _rocm_version_to_package(ver: int) -> str:
    if 400 <= ver < 410:
        # ROCm 4.0
        suffix = '4-0'
    elif 420 <= ver < 430:
        # ROCm 4.2
        suffix = '4-2'
    elif 4_03_00000 <= ver < 4_04_00000:
        # ROCm 4.3
        suffix = '4-3'
    elif 5_00_00000 <= ver < 5_01_00000:
        # ROCm 5.0
        suffix = '5-0'
    else:
        raise AutoDetectionFailed(
            f'Your ROCm version ({ver}) is unsupported.')
    return f'cupy-rocm-{suffix}'


def infer_best_package() -> str:
    """Returns the appropriate CuPy wheel package name for the environment."""

    # Find the existing CuPy wheel installation.
    installed = _find_installed_packages()
    if 1 < len(installed):
        raise AutoDetectionFailed(
            'You have multiple CuPy packages installed: \n'
            f'  {installed}\n'
            'Please uninstall all of them first, then try reinstalling.')

    elif 1 == len(installed):
        if installed[0] in PACKAGES_SDIST:
            raise AutoDetectionFailed(
                'You already have CuPy installed via source'
                ' (pip install cupy).')
        if installed[0] in PACKAGES_OUTDATED:
            raise AutoDetectionFailed(
                f'You have CuPy package "{installed[0]}" installed, but the'
                f' package is not available for version {VERSION}.')
        return installed[0]

    # Try CUDA.
    version = _get_cuda_version()
    if version is not None:
        return _cuda_version_to_package(version)

    # Try ROCm.
    version = _get_rocm_version()
    if version is not None:
        return _rocm_version_to_package(version)

    raise AutoDetectionFailed(
        'Unable to detect NVIDIA CUDA or AMD ROCm installation.')


#
# Entrypoint
#

def main() -> None:
    if os.environ.get('CUPY_UNIVERSAL_PKG_BUILD', None) is None:
        package = infer_best_package()
        requires = f'{package}=={VERSION}'
        _log(f'Installing package: {requires}')
        install_requires = [requires]
    else:
        _log('Building cupy-wheel package for release.')
        install_requires = []

    setup(
        name='cupy-wheel',
        version=VERSION,
        install_requires=install_requires,
    )


if __name__ == '__main__':
    main()
