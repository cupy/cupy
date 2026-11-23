from __future__ import annotations

from distutils import ccompiler
from distutils import sysconfig

from cupy_builder._context import Context
from cupy_builder._features import CUDA_cuda, get_features
from cupy_builder import install_build as build

import cupy
import pytest


def get_compiler_settings():
    ctx = Context('.', _env={}, _argv=[])
    sysconfig.get_config_vars()
    compiler = ccompiler.new_compiler()
    sysconfig.customize_compiler(compiler)
    settings = build.get_compiler_setting(ctx, False)
    return compiler, settings


@pytest.mark.skipif(cupy.cuda.runtime.is_hip, reason='For CUDA environment')
def test_CUDA_cuda():
    ctx = Context('.', _env={}, _argv=[])
    feat = CUDA_cuda(ctx)
    compiler, settings = get_compiler_settings()
    feat.configure(compiler, settings)
    if not cupy.cuda.driver._is_cuda_python():
        # In CUDA Python, `get_build_version()` returns CUDA Python's version
        assert feat._version == cupy.cuda.driver.get_build_version()
    assert feat._version == cupy.cuda.runtime.runtimeGetVersion()


@pytest.mark.parametrize('use_cuda_python', [False, True])
def test_cuda_python_build_requirements(use_cuda_python):
    ctx = Context(
        '.', _env={'CUPY_USE_CUDA_PYTHON': str(int(use_cuda_python))},
        _argv=[])
    features = get_features(ctx)
    cuda, cusolver = features['cuda'], features['cusolver']

    # Always needed, including for device code (curand_kernel.h).
    for header in ('cuda.h', 'cufft.h', 'curand.h'):
        assert header in cuda.includes
    for lib in ('cufft', 'curand'):
        assert lib in cuda.libraries

    # With CUDA Python, these come from nvmath-python at runtime.
    nvmath_includes = {'cublas_v2.h', 'cusparse.h'}
    nvmath_libraries = {'cublas', 'cusparse'}
    if use_cuda_python:
        assert not nvmath_includes & set(cuda.includes)
        assert not nvmath_libraries & set(cuda.libraries)
        assert cusolver.includes == []
        assert cusolver.libraries == []
    else:
        assert nvmath_includes <= set(cuda.includes)
        assert nvmath_libraries <= set(cuda.libraries)
        assert cusolver.includes == ['cusolverDn.h']
        assert cusolver.libraries == ['cusolver']
