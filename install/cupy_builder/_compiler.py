import distutils.ccompiler
import os
import os.path
import platform
import shutil
import sys
import subprocess
from typing import Any, Optional, List

import setuptools
import setuptools.msvc
from setuptools import Extension

from cupy_builder._context import Context
import cupy_builder.install_build as build


def _nvcc_gencode_options(cuda_version: int) -> List[str]:
    """Returns NVCC GPU code generation options."""

    if sys.argv == ['setup.py', 'develop']:
        return []

    envcfg = os.getenv('CUPY_NVCC_GENERATE_CODE', None)
    if envcfg is not None and envcfg != 'current':
        return ['--generate-code={}'.format(arch)
                for arch in envcfg.split(';') if len(arch) > 0]
    if envcfg == 'current' and build.get_compute_capabilities() is not None:
        ccs = build.get_compute_capabilities()
        arch_list = [
            f'compute_{cc}' if cc < 60 else (f'compute_{cc}', f'sm_{cc}')
            for cc in ccs]
    else:
        # The arch_list specifies virtual architectures, such as 'compute_61',
        # and real architectures, such as 'sm_61', for which the CUDA
        # input files are to be compiled.
        #
        # The syntax of an entry of the list is
        #
        #     entry ::= virtual_arch | (virtual_arch, real_arch)
        #
        # where virtual_arch is a string which means a virtual architecture and
        # real_arch is a string which means a real architecture.
        #
        # If a virtual architecture is supplied, NVCC generates a PTX code
        # the virtual architecture. If a pair of a virtual architecture and a
        # real architecture is supplied, NVCC generates a PTX code for the
        # virtual architecture as well as a cubin code for the real one.
        #
        # For example, making NVCC generate a PTX code for 'compute_60' virtual
        # architecture, the arch_list has an entry of 'compute_60'.
        #
        #     arch_list = ['compute_60']
        #
        # For another, making NVCC generate a PTX code for 'compute_61' virtual
        # architecture and a cubin code for 'sm_61' real architecture, the
        # arch_list has an entry of ('compute_61', 'sm_61').
        #
        #     arch_list = [('compute_61', 'sm_61')]
        #
        # See the documentation of each CUDA version for the list of supported
        # architectures:
        #
        #   https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation

        aarch64 = (platform.machine() == 'aarch64')
        if cuda_version >= 11080:
            arch_list = [('compute_35', 'sm_35'),
                         ('compute_37', 'sm_37'),
                         ('compute_50', 'sm_50'),
                         ('compute_52', 'sm_52'),
                         ('compute_60', 'sm_60'),
                         ('compute_61', 'sm_61'),
                         ('compute_70', 'sm_70'),
                         ('compute_75', 'sm_75'),
                         ('compute_80', 'sm_80'),
                         ('compute_86', 'sm_86'),
                         ('compute_89', 'sm_89'),
                         ('compute_90', 'sm_90'),
                         'compute_90']
            if aarch64:
                # Jetson TX1/TX2 are excluded as they don't support JetPack 5
                # (CUDA 11.4).
                arch_list += [
                    # ('compute_53', 'sm_53'),  # Jetson (TX1 / Nano)
                    # ('compute_62', 'sm_62'),  # Jetson (TX2)
                    ('compute_72', 'sm_72'),  # Jetson (Xavier)
                    ('compute_87', 'sm_87'),  # Jetson (Orin)
                ]
        elif cuda_version >= 11040:
            # To utilize CUDA Minor Version Compatibility (`cupy-cuda11x`),
            # CUBIN must be generated for all supported compute capabilities
            # instead of PTX:
            # https://docs.nvidia.com/deploy/cuda-compatibility/index.html#application-considerations
            arch_list = [('compute_35', 'sm_35'),
                         ('compute_37', 'sm_37'),
                         ('compute_50', 'sm_50'),
                         ('compute_52', 'sm_52'),
                         ('compute_60', 'sm_60'),
                         ('compute_61', 'sm_61'),
                         ('compute_70', 'sm_70'),
                         ('compute_75', 'sm_75'),
                         ('compute_80', 'sm_80'),
                         ('compute_86', 'sm_86'),
                         'compute_86']
            if aarch64:
                # Jetson TX1/TX2 are excluded as they don't support JetPack 5
                # (CUDA 11.4).
                arch_list += [
                    # ('compute_53', 'sm_53'),  # Jetson (TX1 / Nano)
                    # ('compute_62', 'sm_62'),  # Jetson (TX2)
                    ('compute_72', 'sm_72'),  # Jetson (Xavier)
                    ('compute_87', 'sm_87'),  # Jetson (Orin)
                ]
        elif cuda_version >= 11010:
            arch_list = ['compute_35',
                         'compute_50',
                         ('compute_60', 'sm_60'),
                         ('compute_61', 'sm_61'),
                         ('compute_70', 'sm_70'),
                         ('compute_75', 'sm_75'),
                         ('compute_80', 'sm_80'),
                         ('compute_86', 'sm_86'),
                         'compute_86']
        elif cuda_version >= 11000:
            arch_list = ['compute_35',
                         'compute_50',
                         ('compute_60', 'sm_60'),
                         ('compute_61', 'sm_61'),
                         ('compute_70', 'sm_70'),
                         ('compute_75', 'sm_75'),
                         ('compute_80', 'sm_80'),
                         'compute_80']
        elif cuda_version >= 10000:
            arch_list = ['compute_30',
                         'compute_50',
                         ('compute_60', 'sm_60'),
                         ('compute_61', 'sm_61'),
                         ('compute_70', 'sm_70'),
                         ('compute_75', 'sm_75'),
                         'compute_70']
        else:
            # This should not happen.
            assert False

    options = []
    for arch in arch_list:
        if type(arch) is tuple:
            virtual_arch, real_arch = arch
            options.append('--generate-code=arch={},code={}'.format(
                virtual_arch, real_arch))
        else:
            options.append('--generate-code=arch={},code={}'.format(
                arch, arch))

    return options


class DeviceCompilerBase:
    """A class that invokes NVCC or HIPCC."""

    def __init__(self, ctx: Context):
        self._context = ctx

    def _get_preprocess_options(self, ext: Extension) -> List[str]:
        # https://setuptools.pypa.io/en/latest/deprecated/distutils/apiref.html#distutils.core.Extension
        # https://github.com/pypa/setuptools/blob/v60.0.0/setuptools/_distutils/command/build_ext.py#L524-L526
        incdirs = ext.include_dirs[:]
        macros: List[Any] = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))
        return distutils.ccompiler.gen_preprocess_options(macros, incdirs)

    def spawn(self, commands: List[str]) -> None:
        print('Command:', commands)
        subprocess.check_call(commands)


class DeviceCompilerUnix(DeviceCompilerBase):

    def compile(self, obj: str, src: str, ext: Extension) -> None:
        if self._context.use_hip:
            self._compile_unix_hipcc(obj, src, ext)
        else:
            self._compile_unix_nvcc(obj, src, ext)

    def _compile_unix_nvcc(self, obj: str, src: str, ext: Extension) -> None:
        cc_args = self._get_preprocess_options(ext) + ['-c']

        # For CUDA C source files, compile them with NVCC.
        nvcc_path = build.get_nvcc_path()
        base_opts = build.get_compiler_base_options(nvcc_path)
        compiler_so = nvcc_path

        cuda_version = self._context.features['cuda'].get_version()
        postargs = _nvcc_gencode_options(cuda_version) + [
            '-O2', '--compiler-options="-fPIC"']
        if cuda_version >= 11020:
            postargs += ['--std=c++14']
            num_threads = int(os.environ.get('CUPY_NUM_NVCC_THREADS', '2'))
            postargs += [f'-t{num_threads}']
        else:
            postargs += ['--std=c++11']
        postargs += ['-Xcompiler=-fno-gnu-unique']
        print('NVCC options:', postargs)
        self.spawn(compiler_so + base_opts + cc_args + [src, '-o', obj] +
                   postargs)

    def _compile_unix_hipcc(self, obj: str, src: str, ext: Extension) -> None:
        cc_args = self._get_preprocess_options(ext) + ['-c']

        # For CUDA C source files, compile them with HIPCC.
        rocm_path = build.get_hipcc_path()
        base_opts = build.get_compiler_base_options(rocm_path)
        compiler_so = rocm_path

        hip_version = build.get_hip_version()
        postargs = ['-O2', '-fPIC', '--include', 'hip_runtime.h']
        if hip_version >= 402:
            postargs += ['--std=c++14']
        else:
            postargs += ['--std=c++11']
        print('HIPCC options:', postargs)
        self.spawn(compiler_so + base_opts + cc_args + [src, '-o', obj] +
                   postargs)


class DeviceCompilerWin32(DeviceCompilerBase):

    def compile(self, obj: str, src: str, ext: Extension) -> None:
        if self._context.use_hip:
            raise RuntimeError('ROCm is not supported on Windows')

        compiler_so = build.get_nvcc_path()
        cc_args = self._get_preprocess_options(ext) + ['-c']
        cuda_version = self._context.features['cuda'].get_version()
        postargs = _nvcc_gencode_options(cuda_version) + ['-O2']
        if cuda_version >= 11020:
            # MSVC 14.0 (2015) is deprecated for CUDA 11.2 but we need it
            # to build CuPy because some Python versions were built using it.
            # REF: https://wiki.python.org/moin/WindowsCompilers
            postargs += ['-allow-unsupported-compiler']
        postargs += ['-Xcompiler', '/MD', '-D_USE_MATH_DEFINES']
        # This is to compile thrust with MSVC2015
        if cuda_version >= 11020:
            postargs += ['--std=c++14']
            num_threads = int(os.environ.get('CUPY_NUM_NVCC_THREADS', '2'))
            postargs += [f'-t{num_threads}']
        cl_exe_path = self._find_host_compiler_path()
        if cl_exe_path is not None:
            print(f'Using host compiler at {cl_exe_path}')
            postargs += ['--compiler-bindir', cl_exe_path]
        print('NVCC options:', postargs)
        self.spawn(compiler_so + cc_args + [src, '-o', obj] + postargs)

    def _find_host_compiler_path(self) -> Optional[str]:
        # c.f. cupy.cuda.compiler._get_extra_path_for_msvc
        cl_exe = shutil.which('cl.exe')
        if cl_exe:
            # The compiler is already on PATH, no extra path needed.
            return None

        vctools: List[str] = setuptools.msvc.EnvironmentInfo(
            platform.machine()).VCTools
        for path in vctools:
            cl_exe = os.path.join(path, 'cl.exe')
            if os.path.exists(cl_exe):
                return path
        print(f'Warning: cl.exe could not be found in {vctools}')
        return None
