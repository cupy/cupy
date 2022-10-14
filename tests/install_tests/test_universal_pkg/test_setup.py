from .. import _from_install_import

import subprocess
import sys

import pytest
import cupy


setup = _from_install_import('universal_pkg.setup')


@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip or cupy.cuda.driver._is_cuda_python(),
    reason='for CUDA')
def test_get_cuda_version():
    assert setup._get_cuda_version() == cupy.cuda.runtime.runtimeGetVersion()


@pytest.mark.skipif(not cupy.cuda.runtime.is_hip, reason='for HIP')
def test_get_rocm_version():
    assert setup._get_rocm_version() == cupy.cuda.runtime.runtimeGetVersion()


def test_cuda_version_to_package():
    with pytest.raises(setup.AutoDetectionFailed):
        assert setup._cuda_version_to_package(10019)
    assert setup._cuda_version_to_package(10020) == 'cupy-cuda102'
    assert setup._cuda_version_to_package(11060) == 'cupy-cuda11x'
    with pytest.raises(setup.AutoDetectionFailed):
        assert setup._cuda_version_to_package(99999)


def test_rocm_version_to_package():
    with pytest.raises(setup.AutoDetectionFailed):
        assert setup._rocm_version_to_package(399)
    assert setup._rocm_version_to_package(4_03_21300) == 'cupy-rocm-4-3'
    assert setup._rocm_version_to_package(5_00_13601) == 'cupy-rocm-5-0'
    with pytest.raises(setup.AutoDetectionFailed):
        assert setup._rocm_version_to_package(9_00_00000)


def test_infer_best_package():
    pkgs = setup._find_installed_packages()
    if 1 < len(pkgs) or pkgs == ['cupy']:
        with pytest.raises(setup.AutoDetectionFailed):
            setup.infer_best_package()
    else:
        assert setup.infer_best_package() == pkgs[0]


def test_execute():
    proc = subprocess.run([sys.executable, setup.__file__, 'help'])
    assert proc.returncode == 1
