import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

import six

from cupy.cuda import _environment
from cupy.cuda import device
from cupy.cuda import function
from cupy.cuda import nvrtc
from cupy.cuda import runtime
from cupy import util

_nvrtc_version = None
_nvrtc_max_compute_capability = None
_win32 = sys.platform.startswith('win32')
_rdc_flags = ('--device-c', '-dc', '-rdc=true',
              '--relocatable-device-code=true')
_cudadevrt = None


class NVCCException(Exception):
    pass


def _run_nvcc(cmd, cwd):
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = ('`nvcc` command returns non-zero exit status. \n'
               'command: {0}\n'
               'return-code: {1}\n'
               'stdout/stderr: \n'
               '{2}'.format(e.cmd,
                            e.returncode,
                            e.output.decode(encoding='UTF-8',
                                            errors='replace')))
        raise NVCCException(msg)
    except OSError as e:
        msg = 'Failed to run `nvcc` command. ' \
              'Check PATH environment variable: ' \
              + str(e)
        raise OSError(msg)


def _get_nvrtc_version():
    global _nvrtc_version
    if _nvrtc_version is None:
        _nvrtc_version = nvrtc.getVersion()

    return _nvrtc_version


def _get_arch():
    global _nvrtc_max_compute_capability
    if _nvrtc_max_compute_capability is None:
        # See Supported Compile Options section of NVRTC User Guide for
        # the maximum value allowed for `--gpu-architecture`.
        major, minor = _get_nvrtc_version()
        if major < 9:
            # CUDA 7.0 / 7.5 / 8.0
            _nvrtc_max_compute_capability = '50'
        else:
            # CUDA 9.0 / 9.1
            _nvrtc_max_compute_capability = '70'

    return min(device.Device().compute_capability,
               _nvrtc_max_compute_capability)


def _is_cudadevrt_needed(options):
    return any(o for o in options if o in _rdc_flags)


def _get_cudadevrt_path():
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


class TemporaryDirectory(object):
    def __enter__(self):
        self.path = tempfile.mkdtemp()
        return self.path

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            return

        for name in os.listdir(self.path):
            os.unlink(os.path.join(self.path, name))
        os.rmdir(self.path)


def _get_bool_env_variable(name, default):
    val = os.environ.get(name)
    if val is None or len(val) == 0:
        return default
    try:
        return int(val) == 1
    except ValueError:
        return False


def compile_using_nvrtc(source, options=(), arch=None, filename='kern.cu'):
    if not arch:
        arch = _get_arch()

    options += ('-arch=compute_{}'.format(arch),)

    with TemporaryDirectory() as root_dir:
        cu_path = os.path.join(root_dir, filename)

        with open(cu_path, 'w') as cu_file:
            cu_file.write(source)

        prog = _NVRTCProgram(source, cu_path)
        try:
            ptx = prog.compile(options)
        except CompileException as e:
            dump = _get_bool_env_variable(
                'CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
            if dump:
                e.dump(sys.stderr)
            raise

        return ptx


def compile_using_nvcc(source, options=(), arch=None,
                       filename='kern.cu', code_type='cubin',
                       separate_compilation=False):
    if not arch:
        arch = _get_arch()

    if code_type not in ('cubin', 'ptx'):
        raise ValueError('Invalid code_type %s. Should be cubin or ptx')
    if code_type == 'ptx':
        assert not separate_compilation

    arch_str = '-gencode=arch=compute_{cc},code=sm_{cc}'.format(cc=arch)
    cmd = [_environment.get_nvcc_path(), arch_str]

    with TemporaryDirectory() as root_dir:
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
                _run_nvcc(cmd, root_dir)
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
                _run_nvcc(cmd, root_dir)
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
                _run_nvcc(cmd, root_dir)
            except NVCCException as e:
                cex = CompileException(str(e), '', '', options, 'nvcc')
                raise cex

        if code_type == 'ptx':
            with open(result_path, 'rb') as ptx_file:
                return ptx_file.read().decode('utf-8')
        elif code_type == 'cubin':
            with open(result_path, 'rb') as bin_file:
                return bin_file.read()
        else:
            assert False, code_type


def _preprocess(source, options, arch, backend):
    if backend == 'nvrtc':
        options += ('-arch=compute_{}'.format(arch),)

        prog = _NVRTCProgram(source, '')
        try:
            result = prog.compile(options)
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

    assert isinstance(result, six.text_type)
    return result


_default_cache_dir = os.path.expanduser('~/.cupy/kernel_cache')


def get_cache_dir():
    return os.environ.get('CUPY_CACHE_DIR', _default_cache_dir)


_empty_file_preprocess_cache = {}


def compile_with_cache(source, options=(), arch=None, cache_dir=None,
                       extra_source=None, backend='nvrtc'):
    if runtime.is_hip:
        return _compile_with_cache_hipcc(source, options, arch, cache_dir,
                                         extra_source)
    else:
        return _compile_with_cache_cuda(source, options, arch, cache_dir,
                                        extra_source, backend)


def _compile_with_cache_cuda(source, options, arch, cache_dir,
                             extra_source=None, backend='nvrtc'):
    # NVRTC does not use extra_source. extra_source is used for cache key.
    global _empty_file_preprocess_cache
    if cache_dir is None:
        cache_dir = get_cache_dir()
    if arch is None:
        arch = _get_arch()

    options += ('-ftz=true',)

    if _get_bool_env_variable('CUPY_CUDA_COMPILE_WITH_DEBUG', False):
        options += ('--device-debug', '--generate-line-info')

    env = (arch, options, _get_nvrtc_version(), backend)
    base = _empty_file_preprocess_cache.get(env, None)
    if base is None:
        # This is checking of NVRTC compiler internal version
        base = _preprocess('', options, arch, backend)
        _empty_file_preprocess_cache[env] = base

    key_src = '%s %s %s %s' % (env, base, source, extra_source)

    key_src = key_src.encode('utf-8')
    name = '%s_2.cubin' % hashlib.md5(key_src).hexdigest()

    if not os.path.isdir(cache_dir):
        try:
            os.makedirs(cache_dir)
        except OSError:
            if not os.path.isdir(cache_dir):
                raise

    mod = function.Module()
    # To handle conflicts in concurrent situation, we adopt lock-free method
    # to avoid performance degradation.
    path = os.path.join(cache_dir, name)
    if os.path.exists(path):
        with open(path, 'rb') as file:
            data = file.read()
        if len(data) >= 32:
            hash = data[:32]
            cubin = data[32:]
            cubin_hash = six.b(hashlib.md5(cubin).hexdigest())
            if hash == cubin_hash:
                mod.load(cubin)
                return mod

    if backend == 'nvrtc':
        ptx = compile_using_nvrtc(source, options, arch, name + '.cu')
        ls = function.LinkState()
        ls.add_ptr_data(ptx, 'cupy.ptx')
        # for separate compilation
        if _is_cudadevrt_needed(options):
            global _cudadevrt
            if _cudadevrt is None:
                _cudadevrt = _get_cudadevrt_path()
            ls.add_ptr_file(_cudadevrt)
        cubin = ls.complete()
    elif backend == 'nvcc':
        rdc = _is_cudadevrt_needed(options)
        cubin = compile_using_nvcc(source, options, arch, name + '.cu',
                                   code_type='cubin', separate_compilation=rdc)
    else:
        raise ValueError('Invalid backend %s' % backend)

    cubin_hash = six.b(hashlib.md5(cubin).hexdigest())

    # shutil.move is not atomic operation, so it could result in a corrupted
    # file. We detect it by appending md5 hash at the beginning of each cache
    # file. If the file is corrupted, it will be ignored next time it is read.
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as tf:
        tf.write(cubin_hash)
        tf.write(cubin)
        temp_path = tf.name
    shutil.move(temp_path, path)

    # Save .cu source file along with .cubin
    if _get_bool_env_variable('CUPY_CACHE_SAVE_CUDA_SOURCE', False):
        with open(path + '.cu', 'w') as f:
            f.write(source)

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
                 include_names=()):
        self.ptr = None

        if isinstance(src, six.binary_type):
            src = src.decode('UTF-8')
        if isinstance(name, six.binary_type):
            name = name.decode('UTF-8')

        self.src = src
        self.name = name
        self.ptr = nvrtc.createProgram(src, name, headers, include_names)

    def __del__(self, is_shutting_down=util.is_shutting_down):
        if is_shutting_down():
            return
        if self.ptr:
            nvrtc.destroyProgram(self.ptr)

    def compile(self, options=()):
        try:
            nvrtc.compileProgram(self.ptr, options)
            return nvrtc.getPTX(self.ptr)
        except nvrtc.NVRTCError:
            log = nvrtc.getProgramLog(self.ptr)
            raise CompileException(log, self.src, self.name, options, 'nvrtc')


def is_valid_kernel_name(name):
    return re.match('^[a-zA-Z_][a-zA-Z_0-9]*$', name) is not None


_hipcc_version = None


def _get_hipcc_version():
    global _hipcc_version
    if _hipcc_version is None:
        cmd = ['hipcc', '--version']
        _hipcc_version = _run_hipcc(cmd)
    return _hipcc_version


def _run_hipcc(cmd, cwd='.', env=None):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, cwd=cwd,
                                       env=env)
    except subprocess.CalledProcessError as e:
        # TODO(leofang): raise an "HIPCCException"?
        raise RuntimeError(
            '`hipcc` command returns non-zero exit status. \n'
            'command: {0}\n'
            'return-code: {1}\n'
            'stdout/stderr: \n'
            '{2}'.format(e.cmd, e.returncode, e.output.decode('utf-8')))
    except OSError as e:
        raise OSError('Failed to run `hipcc` command. '
                      'Check PATH environment variable: '
                      + str(e))


def _hipcc(source, options, arch):
    cmd = ['hipcc', '--genco', '--targets=' + arch,
           '--flags="%s"' % ' '.join(options)]

    with TemporaryDirectory() as root_dir:
        path = os.path.join(root_dir, 'kern')
        in_path = path + '.cpp'
        out_path = path + '.hsaco'

        with open(in_path, 'w') as f:
            f.write(source)

        cmd += [in_path, '-o', out_path]

        env = os.environ.copy()

        output = _run_hipcc(cmd, root_dir, env)
        if not os.path.isfile(out_path):
            raise RuntimeError(
                '`hipcc` command does not generate output file. \n'
                'command: {0}\n'
                'stdout/stderr: \n'
                '{1}'.format(cmd, output))
        with open(out_path, 'rb') as f:
            return f.read()


def _preprocess_hipcc(source, options):
    cmd = ['hipcc', '--preprocess'] + list(options)
    with TemporaryDirectory() as root_dir:
        path = os.path.join(root_dir, 'kern')
        cu_path = '%s.cpp' % path

        with open(cu_path, 'w') as cu_file:
            cu_file.write(source)

        cmd.append(cu_path)
        pp_src = _run_hipcc(cmd, root_dir)
        assert isinstance(pp_src, six.binary_type)
        return re.sub(b'(?m)^#.*$', b'', pp_src)


def _convert_to_hip_source(source):
    table = [
        ('threadIdx.', 'hipThreadIdx_'),
        ('blockIdx.', 'hipBlockIdx_'),
        ('blockDim.', 'hipBlockDim_'),
        ('gridDim.', 'hipGridDim_'),
    ]
    for i, j in table:
        source = source.replace(i, j)

    return "#include <hip/hip_runtime.h>\n" + source


def _compile_with_cache_hipcc(source, options, arch, cache_dir, extra_source,
                              use_converter=True):
    global _empty_file_preprocess_cache
    if cache_dir is None:
        cache_dir = get_cache_dir()
    if arch is None:
        arch = os.environ.get('HCC_AMDGPU_TARGET')
        if arch is None:
            raise RuntimeError('HCC_AMDGPU_TARGET is not set')
    if use_converter:
        source = _convert_to_hip_source(source)

    env = (arch, options, _get_hipcc_version())
    base = _empty_file_preprocess_cache.get(env, None)
    if base is None:
        # This is checking of HIPCC compiler internal version
        base = _preprocess_hipcc('', options)
        _empty_file_preprocess_cache[env] = base
    key_src = '%s %s %s %s' % (env, base, source, extra_source)

    key_src = key_src.encode('utf-8')
    name = '%s.hsaco' % hashlib.md5(key_src).hexdigest()

    if not os.path.isdir(cache_dir):
        try:
            os.makedirs(cache_dir)
        except OSError:
            if not os.path.isdir(cache_dir):
                raise

    mod = function.Module()
    # To handle conflicts in concurrent situation, we adopt lock-free method
    # to avoid performance degradation.
    path = os.path.join(cache_dir, name)
    if os.path.exists(path):
        with open(path, 'rb') as file:
            data = file.read()
        if len(data) >= 32:
            hash_value = data[:32]
            binary = data[32:]
            binary_hash = six.b(hashlib.md5(binary).hexdigest())
            if hash_value == binary_hash:
                mod.load(binary)
                return mod

    # TODO(leofang): catch HIPCCException and convert it to CompileException
    # with backend='hipcc'
    binary = _hipcc(source, options, arch)
    binary_hash = six.b(hashlib.md5(binary).hexdigest())

    # shutil.move is not atomic operation, so it could result in a corrupted
    # file. We detect it by appending md5 hash at the beginning of each cache
    # file. If the file is corrupted, it will be ignored next time it is read.
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as tf:
        tf.write(binary_hash)
        tf.write(binary)
        temp_path = tf.name
    shutil.move(temp_path, path)

    # Save .cu source file along with .hsaco
    if _get_bool_env_variable('CUPY_CACHE_SAVE_CUDA_SOURCE', False):
        with open(path + '.cu', 'w') as f:
            f.write(source)

    mod.load(binary)
    return mod
