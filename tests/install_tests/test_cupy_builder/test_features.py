from distutils import ccompiler
from distutils import sysconfig

from cupy_builder._context import Context
from cupy_builder._features import CUDA_cuda
from cupy_builder import install_build as build

import cupy
import pytest


def get_compiler_settings():
    sysconfig.get_config_vars()
    compiler = ccompiler.new_compiler()
    sysconfig.customize_compiler(compiler)
    settings = build.get_compiler_setting(False)
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
