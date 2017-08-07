import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

import six

from cupy.cuda import device
from cupy.cuda import function


_nvcc_version = None


def _get_nvcc_version():
    global _nvcc_version
    if _nvcc_version is None:
        cmd = ['nvcc', '--version']
        _nvcc_version = _run_nvcc(cmd, '.')

    return _nvcc_version


def _get_arch():
    cc = device.Device().compute_capability
    return 'sm_%s' % cc


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


def _run_nvcc(cmd, cwd):
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = ('`nvcc` command returns non-zero exit status. \n'
               'command: {0}\n'
               'return-code: {1}\n'
               'stdout/stderr: \n'
               '{2}'.format(e.cmd, e.returncode, e.output))
        raise RuntimeError(msg)
    except OSError as e:
        msg = 'Failed to run `nvcc` command. ' \
              'Check PATH environment variable: ' \
              + str(e)
        raise OSError(msg)


def nvcc(source, options=(), arch=None):
    if not arch:
        arch = _get_arch()
    cmd = ['nvcc', '--cubin', '-arch', arch] + list(options)

    with TemporaryDirectory() as root_dir:
        path = os.path.join(root_dir, 'kern')
        cu_path = '%s.cu' % path
        cubin_path = '%s.cubin' % path

        with open(cu_path, 'w') as cu_file:
            cu_file.write(source)

        cmd.append(cu_path)
        _run_nvcc(cmd, root_dir)

        with open(cubin_path, 'rb') as bin_file:
            return bin_file.read()


def preprocess(source, options=()):
    cmd = ['nvcc', '--preprocess'] + list(options)
    with TemporaryDirectory() as root_dir:
        path = os.path.join(root_dir, 'kern')
        cu_path = '%s.cu' % path

        with open(cu_path, 'w') as cu_file:
            cu_file.write(source)

        cmd.append(cu_path)
        pp_src = _run_nvcc(cmd, root_dir)

        if isinstance(pp_src, six.binary_type):
            pp_src = pp_src.decode(sys.getdefaultencoding())
        return re.sub('(?m)^#.*$', '', pp_src)


_default_cache_dir = os.path.expanduser('~/.cupy/kernel_cache')


def get_cache_dir():
    return os.environ.get('CUPY_CACHE_DIR', _default_cache_dir)


_empty_file_preprocess_cache = {}


def compile_with_cache(source, options=(), arch=None, cache_dir=None):
    global _empty_file_preprocess_cache
    if cache_dir is None:
        cache_dir = get_cache_dir()
    if arch is None:
        arch = _get_arch()

    if 'win32' == sys.platform:
        options += ('-Xcompiler', '/wd 4819')
        if sys.maxsize == 9223372036854775807:
            options += '-m64',
        elif sys.maxsize == 2147483647:
            options += '-m32',

    env = (arch, options, _get_nvcc_version())
    if '#include' in source:
        pp_src = '%s %s' % (env, preprocess(source, options))
    else:
        base = _empty_file_preprocess_cache.get(env, None)
        if base is None:
            base = _empty_file_preprocess_cache[env] = preprocess('', options)
        pp_src = '%s %s %s' % (env, base, source)

    if isinstance(pp_src, six.text_type):
        pp_src = pp_src.encode('utf-8')
    name = '%s_2.cubin' % hashlib.md5(pp_src).hexdigest()

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

    cubin = nvcc(source, options, arch)
    cubin_hash = six.b(hashlib.md5(cubin).hexdigest())

    # shutil.move is not atomic operation, so it could result in a corrupted
    # file. We detect it by appending md5 hash at the beginning of each cache
    # file. If the file is corrupted, it will be ignored next time it is read.
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as tf:
        tf.write(cubin_hash)
        tf.write(cubin)
        temp_path = tf.name
    shutil.move(temp_path, path)

    mod.load(cubin)
    return mod
