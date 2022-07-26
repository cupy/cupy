from distutils import ccompiler
from distutils import sysconfig

from cupy_builder._context import Context
from cupy_builder._features import CUDA_cuda
from cupy_builder import install_build as build

import cupy


def get_compiler_settings():
    sysconfig.get_config_vars()
    compiler = ccompiler.new_compiler()
    sysconfig.customize_compiler(compiler)
    settings = build.get_compiler_setting(False)
    return compiler, settings


def test_CUDA_cuda():
    ctx = Context('.', _env={}, _argv=[])
    feat = CUDA_cuda(ctx)
    compiler, settings = get_compiler_settings()
    feat.configure(compiler, settings)
    assert feat._version == cupy.cuda.driver.get_build_version()
