import copy
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

from cupy.cuda import device
from cupy.cuda import function
from cupy_backends.cuda.api import driver
from cupy_backends.cuda.api import runtime
from cupy_backends.cuda.libs import nvrtc
from cupy import _util

if not runtime.is_hip:
    _cuda_version = driver.get_build_version()
    if _cuda_version > 0:
        from cupy.cuda.jitify import jitify


_nvrtc_version = None
_win32 = sys.platform.startswith('win32')
_rdc_flags = ('--device-c', '-dc', '-rdc=true',
              '--relocatable-device-code=true')
_cudadevrt = None


class NVCCException(Exception):
    pass


class HIPCCException(Exception):
    pass


class JitifyException(Exception):
    pass


def _run_cc(cmd, cwd, backend, log_stream=None):
    # backend in ('nvcc', 'hipcc')
    try:
        # Inherit the environment variable as NVCC refers to PATH, TMPDIR/TMP,
        # NVCC_PREPEND_FLAGS, NVCC_APPEND_FLAGS.
        env = os.environ
        if _win32:
            # Adds the extra PATH for NVCC invocation.
            # When running NVCC, a host compiler must be available in PATH,
            # but this is not true in general Windows environment unless
            # running inside the SDK Tools command prompt.
            # To mitigate the situation CuPy automatically adds a path to
            # the VC++ compiler used to build Python / CuPy to the PATH, if
            # VC++ is not available in PATH.
            extra_path = _get_extra_path_for_msvc()
            if extra_path is not None:
                path = extra_path + os.pathsep + os.environ.get('PATH', '')
                env = copy.deepcopy(env)
                env['PATH'] = path
        log = subprocess.check_output(cmd, cwd=cwd, env=env,
                                      stderr=subprocess.STDOUT,
                                      universal_newlines=True)
        if log_stream is not None:
            log_stream.write(log)
        return log
    except subprocess.CalledProcessError as e:
        msg = ('`{0}` command returns non-zero exit status. \n'
               'command: {1}\n'
               'return-code: {2}\n'
               'stdout/stderr: \n'
               '{3}'.format(backend,
                            e.cmd,
                            e.returncode,
                            e.output))
        if backend == 'nvcc':
            raise NVCCException(msg)
        elif backend == 'hipcc':
            raise HIPCCException(msg)
        else:
            raise RuntimeError(msg)
    except OSError as e:
        msg = 'Failed to run `{0}` command. ' \
              'Check PATH environment variable: ' \
              + str(e)
        raise OSError(msg.format(backend))


@_util.memoize()
def _get_extra_path_for_msvc():
    import distutils.spawn
    cl_exe = distutils.spawn.find_executable('cl.exe')
    if cl_exe:
        # The compiler is already on PATH, no extra path needed.
        return None

    from distutils import msvc9compiler
    vcvarsall_bat = msvc9compiler.find_vcvarsall(
        msvc9compiler.get_build_version())
    if not vcvarsall_bat:
        # Failed to find VC.
        return None

    path = os.path.join(os.path.dirname(vcvarsall_bat), 'bin')
    if not distutils.spawn.find_executable('cl.exe', path):
        # The compiler could not be found.
        return None
    return path


def _get_nvrtc_version():
    global _nvrtc_version
    if _nvrtc_version is None:
        _nvrtc_version = nvrtc.getVersion()

    return _nvrtc_version


# Known archs for Tegra/Jetson/Xavier/etc
_tegra_archs = ('53', '62', '72')


@_util.memoize(for_each_device=True)
def _get_arch():
    # See Supported Compile Options section of NVRTC User Guide for
    # the maximum value allowed for `--gpu-architecture`.
    major, minor = _get_nvrtc_version()
    if major < 10 or (major == 10 and minor == 0):
        # CUDA 9.x / 10.0
        _nvrtc_max_compute_capability = '70'
    elif major < 11:
        # CUDA 10.1 / 10.2
        _nvrtc_max_compute_capability = '75'
    else:
        # CUDA 11.0 / 11.1 / 11.2
        _nvrtc_max_compute_capability = '80'

    arch = device.Device().compute_capability
    if arch in _tegra_archs:
        return arch
    else:
        return min(arch, _nvrtc_max_compute_capability)


def _is_cudadevrt_needed(options):
    return any(o for o in options if o in _rdc_flags)


def _get_cudadevrt_path():
    global _cudadevrt
    if _cudadevrt is not None:
        return _cudadevrt

    # defer import to here to avoid circular dependency
    from cupy.cuda import get_cuda_path
    global _win32

    cudadevrt = get_cuda_path()
    if cudadevrt is None:
        raise RuntimeError('CUDA is not found.')

    if _win32:
        # rely on os.altsep
        cudadevrt += '/lib/x64/cudadevrt.lib'
    else:  # linux & osx: search twice as in cupy/install/build.py
        cudadevrt64 = cudadevrt + '/lib64/libcudadevrt.a'
        if not os.path.isfile(cudadevrt64):
            cudadevrt += '/lib/libcudadevrt.a'
        else:
            cudadevrt = cudadevrt64
    if not os.path.isfile(cudadevrt):
        raise RuntimeError(
            'Relocatable PTX code is requested, but cudadevrt '
            'is not found.')
    return cudadevrt


def _remove_rdc_option(options):
    return tuple(o for o in options if o not in _rdc_flags)


def _get_bool_env_variable(name, default):
    val = os.environ.get(name)
    if val is None or len(val) == 0:
        return default
    try:
        return int(val) == 1
    except ValueError:
        return False


_jitify_header_source_map_populated = False


def _jitify_prep(source, options, cu_path):
    # TODO(leofang): refactor this?
    global _jitify_header_source_map_populated
    if not _jitify_header_source_map_populated:
        from cupy._core import core
        _jitify_header_source_map = core._get_header_source_map()
        _jitify_header_source_map_populated = True
    else:
        # this is already cached at the C++ level, so don't pass in anything
        _jitify_header_source_map = None

    # jitify requires the 1st line to be the program name
    old_source = source
    source = cu_path + '\n' + source

    # Upon failure, in addition to throw an error Jitify also prints the log
    # to stdout. In principle we could intercept that by hijacking stdout's
    # file descriptor (tested locally), but the problem is pytest also does
    # the same thing internally, causing strange errors when running the tests.
    # As a result, we currently maintain Jitify's default behavior for easy
    # debugging, and wait for the upstream to address this issue
    # (NVIDIA/jitify#79).

    try:
        name, options, headers, include_names = jitify(
            source, options, _jitify_header_source_map)
    except Exception as e:  # C++ could throw all kinds of errors
        cex = CompileException(str(e), old_source, cu_path, options, 'jitify')
        dump = _get_bool_env_variable(
            'CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
        if dump:
            cex.dump(sys.stderr)
        raise JitifyException(str(cex))
    assert name == cu_path

    return options, headers, include_names


def compile_using_nvrtc(source, options=(), arch=None, filename='kern.cu',
                        name_expressions=None, log_stream=None,
                        cache_in_memory=False, jitify=False):
    # For hipRTC, arch is ignored
    if not runtime.is_hip:
        if not arch:
            arch = _get_arch()
        options += ('-arch=compute_{}'.format(arch),)

    def _compile(
            source, options, cu_path, name_expressions, log_stream, jitify):

        if jitify:
            options, headers, include_names = _jitify_prep(
                source, options, cu_path)
        else:
            headers = include_names = ()

        prog = _NVRTCProgram(source, cu_path, headers, include_names,
                             name_expressions=name_expressions)
        try:
            ptx, mapping = prog.compile(options, log_stream)
        except CompileException as e:
            dump = _get_bool_env_variable(
                'CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
            if dump:
                e.dump(sys.stderr)
            raise
        return ptx, mapping

    if not cache_in_memory:
        with tempfile.TemporaryDirectory() as root_dir:
            cu_path = os.path.join(root_dir, filename)

            with open(cu_path, 'w') as cu_file:
                cu_file.write(source)

            return _compile(source, options, cu_path,
                            name_expressions, log_stream, jitify)
    else:
        cu_path = '' if not jitify else filename
        return _compile(source, options, cu_path, name_expressions,
                        log_stream, jitify)


def compile_using_nvcc(source, options=(), arch=None,
                       filename='kern.cu', code_type='cubin',
                       separate_compilation=False, log_stream=None):
    # defer import to here to avoid circular dependency
    from cupy.cuda import get_nvcc_path

    if not arch:
        arch = _get_arch()

    if code_type not in ('cubin', 'ptx'):
        raise ValueError('Invalid code_type %s. Should be cubin or ptx')
    if code_type == 'ptx':
        assert not separate_compilation

    arch_str = '-gencode=arch=compute_{cc},code=sm_{cc}'.format(cc=arch)
    _nvcc = get_nvcc_path()
    # split() is needed because _nvcc could come from the env var NVCC
    cmd = _nvcc.split()
    cmd.append(arch_str)

    with tempfile.TemporaryDirectory() as root_dir:
        first_part = filename.split('.')[0]

        path = os.path.join(root_dir, first_part)
        cu_path = '%s.cu' % path
        result_path = '%s.%s' % (path, code_type)

        with open(cu_path, 'w') as cu_file:
            cu_file.write(source)

        if not separate_compilation:  # majority cases
            cmd.append('--%s' % code_type)
            cmd += list(options)
            cmd.append(cu_path)

            try:
                _run_cc(cmd, root_dir, 'nvcc', log_stream)
            except NVCCException as e:
                cex = CompileException(str(e), source, cu_path, options,
                                       'nvcc')

                dump = _get_bool_env_variable(
                    'CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
                if dump:
                    cex.dump(sys.stderr)

                raise cex
        else:  # two steps: compile to object and device-link
            cmd_partial = cmd.copy()
            cmd_partial.append('--cubin')

            obj = path + '.o'
            cmd += list(options + ('-o', obj))
            cmd.append(cu_path)

            try:
                _run_cc(cmd, root_dir, 'nvcc', log_stream)
            except NVCCException as e:
                cex = CompileException(str(e), source, cu_path, options,
                                       'nvcc')

                dump = _get_bool_env_variable(
                    'CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
                if dump:
                    cex.dump(sys.stderr)

                raise cex

            options = _remove_rdc_option(options)
            options += ('--device-link', obj, '-o', path + '.cubin')
            cmd = cmd_partial + list(options)

            try:
                _run_cc(cmd, root_dir, 'nvcc', log_stream)
            except NVCCException as e:
                cex = CompileException(str(e), '', '', options, 'nvcc')
                raise cex

        if code_type == 'ptx':
            with open(result_path, 'rb') as ptx_file:
                return ptx_file.read()
        elif code_type == 'cubin':
            with open(result_path, 'rb') as bin_file:
                return bin_file.read()
        else:
            assert False, code_type


def _preprocess(source, options, arch, backend):
    if backend == 'nvrtc':
        options += ('-arch=compute_{}'.format(arch),)

        prog = _NVRTCProgram(source)
        try:
            result, _ = prog.compile(options)
        except CompileException as e:
            dump = _get_bool_env_variable(
                'CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
            if dump:
                e.dump(sys.stderr)
            raise
    elif backend == 'nvcc':
        try:
            result = compile_using_nvcc(source, options, arch, 'preprocess.cu',
                                        code_type='ptx')
        except CompileException as e:
            dump = _get_bool_env_variable(
                'CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
            if dump:
                e.dump(sys.stderr)
            raise
    else:
        raise ValueError('Invalid backend %s' % backend)

    assert isinstance(result, bytes)
    return result


_default_cache_dir = os.path.expanduser('~/.cupy/kernel_cache')


def get_cache_dir():
    return os.environ.get('CUPY_CACHE_DIR', _default_cache_dir)


_empty_file_preprocess_cache = {}


def compile_with_cache(
        source, options=(), arch=None, cache_dir=None, extra_source=None,
        backend='nvrtc', *, enable_cooperative_groups=False,
        name_expressions=None, log_stream=None, jitify=False):

    if enable_cooperative_groups:
        if runtime.is_hip:
            raise ValueError(
                'Cooperative groups is not supported in HIP.')

    if name_expressions is not None and backend != 'nvrtc':
        raise NotImplementedError

    # We silently ignore CUPY_CACHE_IN_MEMORY if nvcc/hipcc are in use, because
    # they must dump files to disk.
    cache_in_memory = (
        _get_bool_env_variable('CUPY_CACHE_IN_MEMORY', False)
        and backend == 'nvrtc')

    if runtime.is_hip:
        backend = 'hiprtc' if backend == 'nvrtc' else 'hipcc'
        return _compile_with_cache_hip(
            source, options, arch, cache_dir, extra_source, backend,
            name_expressions, log_stream, cache_in_memory)
    else:
        return _compile_with_cache_cuda(
            source, options, arch, cache_dir, extra_source, backend,
            enable_cooperative_groups, name_expressions, log_stream,
            cache_in_memory, jitify)


def _compile_with_cache_cuda(
        source, options, arch, cache_dir, extra_source=None, backend='nvrtc',
        enable_cooperative_groups=False, name_expressions=None,
        log_stream=None, cache_in_memory=False, jitify=False):
    # NVRTC does not use extra_source. extra_source is used for cache key.
    global _empty_file_preprocess_cache
    if cache_dir is None:
        cache_dir = get_cache_dir()
    if arch is None:
        arch = _get_arch()

    options += ('-ftz=true',)

    if enable_cooperative_groups:
        # `cooperative_groups` requires relocatable device code.
        options += ('--device-c',)

    if _get_bool_env_variable('CUPY_CUDA_COMPILE_WITH_DEBUG', False):
        options += ('--device-debug', '--generate-line-info')

    is_jitify_requested = ('-DCUPY_USE_JITIFY' in options)
    if jitify and not is_jitify_requested:
        # jitify is set in RawKernel/RawModule, translate it to an option
        # that is useless to the compiler, but can be used as part of the
        # hash key
        options += ('-DCUPY_USE_JITIFY',)
    elif is_jitify_requested and not jitify:
        # jitify is requested internally, just set the flag
        jitify = True
    if jitify and backend != 'nvrtc':
        raise ValueError('jitify only works with NVRTC')

    env = (arch, options, _get_nvrtc_version(), backend)
    base = _empty_file_preprocess_cache.get(env, None)
    if base is None:
        # This is for checking NVRTC/NVCC compiler internal version
        base = _preprocess('', options, arch, backend)
        _empty_file_preprocess_cache[env] = base

    key_src = '%s %s %s %s' % (env, base, source, extra_source)
    key_src = key_src.encode('utf-8')
    name = '%s_2.cubin' % hashlib.md5(key_src).hexdigest()

    mod = function.Module()

    if not cache_in_memory:
        # Read from disk cache
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        # To handle conflicts in concurrent situation, we adopt lock-free
        # method to avoid performance degradation.
        # We force recompiling to retrieve C++ mangled names if so desired.
        path = os.path.join(cache_dir, name)
        if os.path.exists(path) and not name_expressions:
            with open(path, 'rb') as file:
                data = file.read()
            if len(data) >= 32:
                hash = data[:32]
                cubin = data[32:]
                cubin_hash = hashlib.md5(cubin).hexdigest().encode('ascii')
                if hash == cubin_hash:
                    mod.load(cubin)
                    return mod
    else:
        # Enforce compiling -- the resulting kernel will be cached elsewhere,
        # so we do nothing
        pass

    if backend == 'nvrtc':
        cu_name = '' if cache_in_memory else name + '.cu'
        ptx, mapping = compile_using_nvrtc(
            source, options, arch, cu_name, name_expressions,
            log_stream, cache_in_memory, jitify)
        if _is_cudadevrt_needed(options):
            # for separate compilation
            ls = function.LinkState()
            ls.add_ptr_data(ptx, 'cupy.ptx')
            _cudadevrt = _get_cudadevrt_path()
            ls.add_ptr_file(_cudadevrt)
            cubin = ls.complete()
        else:
            cubin = ptx
        mod._set_mapping(mapping)
    elif backend == 'nvcc':
        rdc = _is_cudadevrt_needed(options)
        cubin = compile_using_nvcc(source, options, arch,
                                   name + '.cu', code_type='cubin',
                                   separate_compilation=rdc,
                                   log_stream=log_stream)
    else:
        raise ValueError('Invalid backend %s' % backend)

    if not cache_in_memory:
        # Write to disk cache
        cubin_hash = hashlib.md5(cubin).hexdigest().encode('ascii')

        # shutil.move is not atomic operation, so it could result in a
        # corrupted file. We detect it by appending md5 hash at the beginning
        # of each cache file. If the file is corrupted, it will be ignored
        # next time it is read.
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as tf:
            tf.write(cubin_hash)
            tf.write(cubin)
            temp_path = tf.name
        shutil.move(temp_path, path)

        # Save .cu source file along with .cubin
        if _get_bool_env_variable('CUPY_CACHE_SAVE_CUDA_SOURCE', False):
            with open(path + '.cu', 'w') as f:
                f.write(source)
    else:
        # we don't do any disk I/O
        pass

    mod.load(cubin)
    return mod


class CompileException(Exception):

    def __init__(self, msg, source, name, options, backend='nvrtc'):
        self._msg = msg
        self.source = source
        self.name = name
        self.options = options
        self.backend = backend
        super(CompileException, self).__init__()

    def __reduce__(self):
        return (type(self), (self._msg, self.source, self.name,
                             self.options, self.backend))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.get_message()

    def get_message(self):
        return self._msg

    def dump(self, f):
        lines = self.source.split('\n')
        digits = int(math.floor(math.log10(len(lines)))) + 1
        linum_fmt = '{{:0{}d}} '.format(digits)
        f.write('{} '.format(self.backend.upper()))
        f.write('compilation error: {}\n'.format(self))
        f.write('-----\n')
        f.write('Name: {}\n'.format(self.name))
        f.write('Options: {}\n'.format(' '.join(self.options)))
        f.write('CUDA source:\n')
        for i, line in enumerate(lines):
            f.write(linum_fmt.format(i + 1) + line.rstrip() + '\n')
        f.write('-----\n')
        f.flush()


class _NVRTCProgram(object):

    def __init__(self, src, name='default_program', headers=(),
                 include_names=(), name_expressions=None):
        self.ptr = None

        if isinstance(src, bytes):
            src = src.decode('UTF-8')
        if isinstance(name, bytes):
            name = name.decode('UTF-8')

        self.src = src
        self.name = name
        self.ptr = nvrtc.createProgram(src, name, headers, include_names)
        self.name_expressions = name_expressions

    def __del__(self, is_shutting_down=_util.is_shutting_down):
        if is_shutting_down():
            return
        if self.ptr:
            nvrtc.destroyProgram(self.ptr)

    def compile(self, options=(), log_stream=None):
        try:
            if self.name_expressions:
                for ker in self.name_expressions:
                    nvrtc.addAddNameExpression(self.ptr, ker)
            nvrtc.compileProgram(self.ptr, options)
            mapping = None
            if self.name_expressions:
                mapping = {}
                for ker in self.name_expressions:
                    mapping[ker] = nvrtc.getLoweredName(self.ptr, ker)
            if log_stream is not None:
                log_stream.write(nvrtc.getProgramLog(self.ptr))
            # TODO(leofang): use getCUBIN() for _cuda_version >= 11010?
            return nvrtc.getPTX(self.ptr), mapping
        except nvrtc.NVRTCError:
            log = nvrtc.getProgramLog(self.ptr)
            raise CompileException(log, self.src, self.name, options,
                                   'nvrtc' if not runtime.is_hip else 'hiprtc')


def is_valid_kernel_name(name):
    return re.match('^[a-zA-Z_][a-zA-Z_0-9]*$', name) is not None


def compile_using_hipcc(source, options, arch, log_stream=None):
    # As of ROCm 3.5.0 hiprtc/hipcc can automatically pick up the
    # right arch without setting HCC_AMDGPU_TARGET, so we don't need
    # to set arch here
    cmd = ['hipcc', '--genco'] + list(options)

    with tempfile.TemporaryDirectory() as root_dir:
        path = os.path.join(root_dir, 'kern')
        in_path = path + '.cpp'
        out_path = path + '.hsaco'

        with open(in_path, 'w') as f:
            f.write(source)

        cmd += [in_path, '-o', out_path]

        try:
            output = _run_cc(cmd, root_dir, 'hipcc', log_stream)
        except HIPCCException as e:
            cex = CompileException(str(e), source, in_path, options,
                                   'hipcc')

            dump = _get_bool_env_variable(
                'CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
            if dump:
                cex.dump(sys.stderr)

            raise cex
        if not os.path.isfile(out_path):
            raise HIPCCException(
                '`hipcc` command does not generate output file. \n'
                'command: {0}\n'
                'stdout/stderr: \n'
                '{1}'.format(cmd, output))
        with open(out_path, 'rb') as f:
            return f.read()


# TODO(leofang): consider merge _preprocess_hipcc with _preprocess_hiprtc,
# perhaps also with _preprocess?
def _preprocess_hipcc(source, options):
    cmd = ['hipcc', '--preprocess'] + list(options)
    with tempfile.TemporaryDirectory() as root_dir:
        path = os.path.join(root_dir, 'kern')
        cu_path = '%s.cpp' % path

        with open(cu_path, 'w') as cu_file:
            cu_file.write(source)

        cmd.append(cu_path)
        pp_src = _run_cc(cmd, root_dir, 'hipcc')
        assert isinstance(pp_src, str)
        return re.sub('(?m)^#.*$', '', pp_src)


def _preprocess_hiprtc(source, options):
    # source is ignored
    prog = _NVRTCProgram(
        '''
        // hiprtc segfaults if the input code is empty
        #include <hip/hip_runtime.h>
        __global__ void _cupy_preprocess_dummy_kernel_() { }
        ''')
    try:
        result, _ = prog.compile(options)
    except CompileException as e:
        dump = _get_bool_env_variable(
            'CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
        if dump:
            e.dump(sys.stderr)
        raise
    assert isinstance(result, bytes)
    return result


_hip_extra_source = None


def _convert_to_hip_source(source, extra_source, is_hiprtc):
    if not is_hiprtc:
        return '#include <hip/hip_runtime.h>\n' + source

    # Workaround for hiprtc: it does not follow the -I option to search
    # headers (as of ROCm 3.5.0), so we must prepend all CuPy's headers
    global _hip_extra_source
    if _hip_extra_source is None:
        if extra_source is not None:
            extra_source = extra_source.split('\n')
            extra_source = [line for line in extra_source if (
                not line.startswith('#include')
                and not line.startswith('#pragma once'))]
            _hip_extra_source = extra_source = '\n'.join(extra_source)

    source = source.split('\n')
    source = [line for line in source if not line.startswith('#include')]
    source = ('#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n'
              + _hip_extra_source + '\n'.join(source))

    return source


# TODO(leofang): evaluate if this can be merged with _compile_with_cache_cuda()
def _compile_with_cache_hip(source, options, arch, cache_dir, extra_source,
                            backend='hiprtc', name_expressions=None,
                            log_stream=None, cache_in_memory=False,
                            use_converter=True):
    global _empty_file_preprocess_cache

    # TODO(leofang): this might be possible but is currently undocumented
    if _is_cudadevrt_needed(options):
        raise ValueError('separate compilation is not supported in HIP')

    # HIP's equivalent of -ftz=true, see ROCm-Developer-Tools/HIP#2252
    # Notes:
    # - For hipcc, this should just work, as invalid options would cause errors
    #   See https://clang.llvm.org/docs/ClangCommandLineReference.html.
    # - For hiprtc, this is a no-op until the compiler options like -D and -I
    #   are accepted, see ROCm-Developer-Tools/HIP#2182 and
    #   ROCm-Developer-Tools/HIP#2248
    options += ('-fcuda-flush-denormals-to-zero',)

    if cache_dir is None:
        cache_dir = get_cache_dir()
    # As of ROCm 3.5.0 hiprtc/hipcc can automatically pick up the
    # right arch without setting HCC_AMDGPU_TARGET, so we don't need
    # to tell the compiler which arch we are targeting. But, we still
    # need to know arch as part of the cache key:
    if arch is None:
        # On HIP, gcnArch is computed from "compute capability":
        # https://github.com/ROCm-Developer-Tools/HIP/blob/rocm-4.0.0/rocclr/hip_device.cpp#L202
        arch = device.Device().compute_capability
    if use_converter:
        source = _convert_to_hip_source(source, extra_source,
                                        is_hiprtc=(backend == 'hiprtc'))

    env = (arch, options, _get_nvrtc_version(), backend)
    base = _empty_file_preprocess_cache.get(env, None)
    if base is None:
        # This is for checking HIPRTC/HIPCC compiler internal version
        if backend == 'hiprtc':
            base = _preprocess_hiprtc('', options)
        else:
            base = _preprocess_hipcc('', options)
        _empty_file_preprocess_cache[env] = base

    key_src = '%s %s %s %s' % (env, base, source, extra_source)
    key_src = key_src.encode('utf-8')
    name = '%s.hsaco' % hashlib.md5(key_src).hexdigest()

    mod = function.Module()

    if not cache_in_memory:
        # Read from disk cache
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        # To handle conflicts in concurrent situation, we adopt lock-free
        # method to avoid performance degradation.
        # We force recompiling to retrieve C++ mangled names if so desired.
        path = os.path.join(cache_dir, name)
        if os.path.exists(path) and not name_expressions:
            with open(path, 'rb') as f:
                data = f.read()
            if len(data) >= 32:
                hash_value = data[:32]
                binary = data[32:]
                binary_hash = hashlib.md5(binary).hexdigest().encode('ascii')
                if hash_value == binary_hash:
                    mod.load(binary)
                    return mod
    else:
        # Enforce compiling -- the resulting kernel will be cached elsewhere,
        # so we do nothing
        pass

    if backend == 'hiprtc':
        # compile_using_nvrtc calls hiprtc for hip builds
        binary, mapping = compile_using_nvrtc(
            source, options, arch, name + '.cu', name_expressions,
            log_stream, cache_in_memory)
        mod._set_mapping(mapping)
    else:
        binary = compile_using_hipcc(source, options, arch, log_stream)

    if not cache_in_memory:
        # Write to disk cache
        binary_hash = hashlib.md5(binary).hexdigest().encode('ascii')

        # shutil.move is not atomic operation, so it could result in a
        # corrupted file. We detect it by appending md5 hash at the beginning
        # of each cache file. If the file is corrupted, it will be ignored
        # next time it is read.
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as tf:
            tf.write(binary_hash)
            tf.write(binary)
            temp_path = tf.name
        shutil.move(temp_path, path)

        # Save .cu source file along with .hsaco
        if _get_bool_env_variable('CUPY_CACHE_SAVE_CUDA_SOURCE', False):
            with open(path + '.cpp', 'w') as f:
                f.write(source)
    else:
        # we don't do any disk I/O
        pass

    mod.load(binary)
    return mod
