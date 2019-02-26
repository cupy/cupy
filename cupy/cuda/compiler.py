import hashlib
import math
import os
import re
import shutil
import sys
import tempfile

import six

from cupy.cuda import device
from cupy.cuda import function
from cupy.cuda import nvrtc

_nvrtc_version = None
_nvrtc_max_compute_capability = None


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
    cc = min(device.Device().compute_capability, _nvrtc_max_compute_capability)
    return 'compute_%s' % cc


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

    options += ('-arch={}'.format(arch),)

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


def _preprocess(source, options, arch):
    options += ('-arch={}'.format(arch),)

    prog = _NVRTCProgram(source, '')
    try:
        result = prog.compile(options)
    except CompileException as e:
        dump = _get_bool_env_variable(
            'CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
        if dump:
            e.dump(sys.stderr)
        raise

    assert isinstance(result, six.text_type)
    return result


_default_cache_dir = os.path.expanduser('~/.cupy/kernel_cache')


def get_cache_dir():
    return os.environ.get('CUPY_CACHE_DIR', _default_cache_dir)


_empty_file_preprocess_cache = {}


def compile_with_cache(source, options=(), arch=None, cache_dir=None,
                       extra_source=None):
    # NVRTC does not use extra_source. extra_source is used for cache key.
    global _empty_file_preprocess_cache
    if cache_dir is None:
        cache_dir = get_cache_dir()
    if arch is None:
        arch = _get_arch()

    options += ('-ftz=true',)
    if _get_bool_env_variable('CUPY_CUDA_COMPILE_WITH_DEBUG', False):
        options += ('--device-debug', '--generate-line-info')

    env = (arch, options, _get_nvrtc_version())
    base = _empty_file_preprocess_cache.get(env, None)
    if base is None:
        # This is checking of NVRTC compiler internal version
        base = _preprocess('', options, arch)
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

    ptx = compile_using_nvrtc(source, options, arch, name + '.cu')
    ls = function.LinkState()
    ls.add_ptr_data(ptx, u'cupy.ptx')
    cubin = ls.complete()
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

    def __init__(self, msg, source, name, options):
        self._msg = msg
        self.source = source
        self.name = name
        self.options = options

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
        f.write('NVRTC compilation error: {}\n'.format(self))
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

    def __del__(self):
        if self.ptr:
            nvrtc.destroyProgram(self.ptr)

    def compile(self, options=()):
        try:
            nvrtc.compileProgram(self.ptr, options)
            return nvrtc.getPTX(self.ptr)
        except nvrtc.NVRTCError:
            log = nvrtc.getProgramLog(self.ptr)
            raise CompileException(log, self.src, self.name, options)


def is_valid_kernel_name(name):
    return re.match('^[a-zA-Z_][a-zA-Z_0-9]*$', name) is not None
