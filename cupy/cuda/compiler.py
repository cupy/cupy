import fcntl
import hashlib
import os
import re
import subprocess
import tempfile

import six

from cupy.cuda import device
from cupy.cuda import module


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


def nvcc(source, options=None, arch=None):
    cmd = ['nvcc', '--cubin']
    if options:
        cmd += options
    if not arch:
        arch = _get_arch()
    cmd += ['-arch', arch]

    with TemporaryDirectory() as root_dir:
        path = os.path.join(root_dir, 'kern')
        cu_path = '%s.cu' % path
        cubin_path = '%s.cubin' % path

        with open(cu_path, 'w') as cu_file:
            cu_file.write(source)

        cmd.append(cu_path)
        subprocess.check_call(cmd, cwd=root_dir)

        with open(cubin_path, 'rb') as bin_file:
            return bin_file.read()


def preprocess(source, options=()):
    cmd = ['nvcc', '--preprocess']
    if options is not None:
        cmd += list(options)
    with TemporaryDirectory() as root_dir:
        path = os.path.join(root_dir, 'kern')
        cu_path = '%s.cu' % path

        with open(cu_path, 'w') as cu_file:
            cu_file.write(source)

        cmd.append(cu_path)
        pp_src = subprocess.check_output(cmd, cwd=root_dir)
        if isinstance(pp_src, six.binary_type):
            pp_src = pp_src.decode('utf-8')
        return re.sub('(?m)^#.*$', '', pp_src)


_default_cache_dir = os.path.expanduser('~/.cupy/kernel_cache')


def get_cache_dir():
    return os.environ.get('CUPY_CACHE_DIR', _default_cache_dir)


def compile_with_cache(source, options=[], arch=None, cache_dir=None):
    if cache_dir is None:
        cache_dir = get_cache_dir()
    if arch is None:
        arch = _get_arch()

    pp_src = '%s %s' % (arch, preprocess(source, options))
    if isinstance(pp_src, six.text_type):
        pp_src = pp_src.encode('utf-8')
    name = '%s.cubin' % hashlib.md5(pp_src).hexdigest()

    mod = module.Module()

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    lock_path = os.path.join(cache_dir, 'lock_file.lock')
    if not os.path.exists(lock_path):
        open(lock_path, 'a').close()

    path = os.path.join(cache_dir, name)
    with open(lock_path, 'r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(path):
            with open(path, 'rb') as file:
                cubin = file.read()
            mod.load(cubin)
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
            cubin = nvcc(source, options, arch)
            mod.load(cubin)
            fcntl.flock(lock, fcntl.LOCK_EX)
            with open(path, 'wb') as cubin_file:
                cubin_file.write(cubin)
        fcntl.flock(lock, fcntl.LOCK_UN)

    return mod
