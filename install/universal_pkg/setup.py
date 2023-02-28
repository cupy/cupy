import ctypes
import pkg_resources
import os
import sys
from typing import Dict, List, Optional

from setuptools import setup


VERSION = '12.0.0rc1'

# List of packages supported by this version of CuPy.
PACKAGES = [
    'cupy-cuda102',
    'cupy-cuda110',
    'cupy-cuda111',
    'cupy-cuda11x',
    'cupy-cuda12x',
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
    'cupy-cuda112',
    'cupy-cuda113',
    'cupy-cuda114',
    'cupy-cuda115',
    'cupy-cuda116',
    'cupy-cuda117',
    'cupy-rocm-4-0',
    'cupy-rocm-4-2',
]

# List of sdist packages.
PACKAGES_SDIST = [
    'cupy',
]


class AutoDetectionFailed(Exception):
    def __str__(self) -> str:
        return f'''
============================================================
{super().__str__()}
============================================================
'''


def _log(msg: str) -> None:
    sys.stdout.write(f'[cupy-wheel] {msg}\n')
    sys.stdout.flush()


def _get_version_from_library(
        libnames: List[str],
        funcname: str,
        nvrtc: bool = False,
) -> Optional[int]:
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
    func.restype = ctypes.c_int

    if nvrtc:
        # nvrtcVersion
        func.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        major = ctypes.c_int()
        minor = ctypes.c_int()
        retval = func(major, minor)
        version = major.value * 1000 + minor.value * 10
    else:
        # cudaRuntimeGetVersion
        func.argtypes = [
            ctypes.POINTER(ctypes.c_int),
        ]
        version_ref = ctypes.c_int()
        retval = func(version_ref)
        version = version_ref.value

    if retval != 0:  # NVRTC_SUCCESS or cudaSuccess
        raise AutoDetectionFailed(
            f'{libname}: {func} returned error: {retval}')
    _log(f'Detected version: {version}')
    return version


def _setup_win32_dll_directory() -> None:
    if not hasattr(os, 'add_dll_directory'):
        # Python 3.7 or earlier.
        return
    cuda_path = os.environ.get('CUDA_PATH', None)
    if cuda_path is None:
        _log('CUDA_PATH is not set.'
             'cupy-wheel may not be able to discover NVRTC to probe version')
        return
    os.add_dll_directory(os.path.join(cuda_path, 'bin'))  # type: ignore[attr-defined] # NOQA


def _get_cuda_version() -> Optional[int]:
    """Returns the detected CUDA version or None."""

    if sys.platform == 'linux':
        libnames = [
            'libnvrtc.so.12',
            'libnvrtc.so.11.2',
            'libnvrtc.so.11.1',
            'libnvrtc.so.11.0',
            'libnvrtc.so.10.2',
        ]
    elif sys.platform == 'win32':
        libnames = [
            'nvrtc64_120_0.dll',
            'nvrtc64_112_0.dll',
            'nvrtc64_111_0.dll',
            'nvrtc64_110_0.dll',
            'nvrtc64_102_0.dll',
        ]
        _setup_win32_dll_directory()
    else:
        _log(f'CUDA detection unsupported on platform: {sys.platform}')
        return None
    _log(f'Trying to detect CUDA version from libraries: {libnames}')
    version = _get_version_from_library(libnames, 'nvrtcVersion', True)
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
    elif ver < 12000:
        # CUDA 11.2 ~ 11.x
        suffix = '11x'
    elif ver < 13000:
        # CUDA 12.x
        suffix = '12x'
    else:
        raise AutoDetectionFailed(
            f'Your CUDA version ({ver}) is too new.')
    return f'cupy-cuda{suffix}'


def _rocm_version_to_package(ver: int) -> str:
    """
    ROCm 4.0.x = 3212
    ROCm 4.1.x = 3241
    ROCm 4.2.0 = 3275
    ROCm 4.3.0 = 40321300
    ROCm 4.3.1 = 40321331
    ROCm 4.5.0 = 40421401
    ROCm 4.5.1 = 40421432
    ROCm 5.0.0 = 50013601
    ROCm 5.1.0 = 50120531
    """
    if 4_03_00000 <= ver < 4_04_00000:
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
                f' package is not available for version {VERSION}.\n'
                'Hint: cupy-cuda{112~117} has been merged to cupy-cuda11x in '
                'CuPy v11. Uninstall the package and try again.')
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


def _get_cmdclass(tag: str) -> Dict[str, type]:
    try:
        import wheel.bdist_wheel
    except ModuleNotFoundError:
        return {}

    class bdist_wheel_with_tag(wheel.bdist_wheel.bdist_wheel):  # type: ignore[misc] # NOQA
        def initialize_options(self) -> None:
            super().initialize_options()
            self.build_number = f'0_{tag}'

    return {"bdist_wheel": bdist_wheel_with_tag}


#
# Entrypoint
#

def main() -> None:
    if os.environ.get('CUPY_UNIVERSAL_PKG_BUILD', None) is None:
        package = infer_best_package()
        requires = f'{package}=={VERSION}'
        _log(f'Installing package: {requires}')
        install_requires = [requires]
        tag = package
    else:
        _log('Building cupy-wheel package for release.')
        install_requires = []
        tag = '0'

    setup(
        name='cupy-wheel',
        version=f'{VERSION}',
        install_requires=install_requires,
        cmdclass=_get_cmdclass(tag),
    )


if __name__ == '__main__':
    main()
