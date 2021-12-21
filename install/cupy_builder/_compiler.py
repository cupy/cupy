def _nvcc_gencode_options(cuda_version):
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

        if cuda_version >= 11040:
            arch_list = ['compute_35',
                         'compute_50',
                         ('compute_60', 'sm_60'),
                         ('compute_61', 'sm_61'),
                         ('compute_70', 'sm_70'),
                         ('compute_75', 'sm_75'),
                         ('compute_80', 'sm_80'),
                         ('compute_86', 'sm_86'),
                         ('compute_87', 'sm_87'),
                         'compute_87']
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


class _UnixCCompiler(unixccompiler.UnixCCompiler):
    src_extensions = list(unixccompiler.UnixCCompiler.src_extensions)
    src_extensions.append('.cu')

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        # For sources other than CUDA C ones, just call the super class method.
        if os.path.splitext(src)[1] != '.cu':
            return unixccompiler.UnixCCompiler._compile(
                self, obj, src, ext, cc_args, extra_postargs, pp_opts)

        if use_hip:
            return self._compile_unix_hipcc(
                obj, src, ext, cc_args, extra_postargs, pp_opts)
        else:
            return self._compile_unix_nvcc(
                obj, src, ext, cc_args, extra_postargs, pp_opts)

    def _compile_unix_nvcc(self,
                           obj, src, ext, cc_args, extra_postargs, pp_opts):
        # For CUDA C source files, compile them with NVCC.
        nvcc_path = build.get_nvcc_path()
        base_opts = build.get_compiler_base_options(nvcc_path)
        compiler_so = nvcc_path

        cuda_version = build.get_cuda_version()
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
        try:
            self.spawn(compiler_so + base_opts + cc_args + [src, '-o', obj] +
                       postargs)
        except errors.DistutilsExecError as e:
            raise errors.CompileError(str(e))

    def _compile_unix_hipcc(self,
                            obj, src, ext, cc_args, extra_postargs, pp_opts):
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
        try:
            self.spawn(compiler_so + base_opts + cc_args + [src, '-o', obj] +
                       postargs)
        except errors.DistutilsExecError as e:
            raise errors.CompileError(str(e))

    def link(self, target_desc, objects, output_filename, *args):
        use_hipcc = False
        if use_hip:
            for i in objects:
                if any(obj in i for obj in ('cupy_thrust.o', 'cupy_cub.o')):
                    use_hipcc = True
        if use_hipcc:
            _compiler_cxx = self.compiler_cxx
            try:
                rocm_path = build.get_hipcc_path()
                self.set_executable('compiler_cxx', rocm_path)

                return unixccompiler.UnixCCompiler.link(
                    self, target_desc, objects, output_filename, *args)
            finally:
                self.compiler_cxx = _compiler_cxx
        else:
            return unixccompiler.UnixCCompiler.link(
                self, target_desc, objects, output_filename, *args)


class _MSVCCompiler(msvccompiler.MSVCCompiler):
    _cu_extensions = ['.cu']

    src_extensions = list(unixccompiler.UnixCCompiler.src_extensions)
    src_extensions.extend(_cu_extensions)

    def _compile_cu(self, sources, output_dir=None, macros=None,
                    include_dirs=None, debug=0, extra_preargs=None,
                    extra_postargs=None, depends=None):
        # Compile CUDA C files, mainly derived from UnixCCompiler._compile().
        macros, objects, extra_postargs, pp_opts, _build = \
            self._setup_compile(output_dir, macros, include_dirs, sources,
                                depends, extra_postargs)

        compiler_so = build.get_nvcc_path()
        cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
        cuda_version = build.get_cuda_version()
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
        print('NVCC options:', postargs)

        for obj in objects:
            try:
                src, ext = _build[obj]
            except KeyError:
                continue
            try:
                self.spawn(compiler_so + cc_args + [src, '-o', obj] + postargs)
            except errors.DistutilsExecError as e:
                raise errors.CompileError(str(e))

        return objects

    def compile(self, sources, **kwargs):
        # Split CUDA C sources and others.
        if use_hip:
            raise RuntimeError('ROCm is not supported on Windows')

        cu_sources = []
        other_sources = []
        for source in sources:
            if os.path.splitext(source)[1] == '.cu':
                cu_sources.append(source)
            else:
                other_sources.append(source)

        # Compile source files other than CUDA C ones.
        other_objects = msvccompiler.MSVCCompiler.compile(
            self, other_sources, **kwargs)

        # Compile CUDA C sources.
        cu_objects = self._compile_cu(cu_sources, **kwargs)

        # Return compiled object filenames.
        return other_objects + cu_objects
