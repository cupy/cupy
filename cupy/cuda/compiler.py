import hashlib
import os
import re
import tempfile

import filelock
import six

from cupy.cuda import device
from cupy.cuda import function
from pynvrtc.compiler import NVRTCInterface
from pynvrtc.compiler import Program

_nvrtc_version = None


def _get_nvrtc_version():
    global _nvrtc_version
    if _nvrtc_version is None:
        interface = NVRTCInterface()
        _nvrtc_version = interface.nvrtcVersion()

    return _nvrtc_version


def _get_arch():
    cc = device.Device().compute_capability
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


def nvrtc(source, options=(), arch=None):
    if not arch:
        arch = _get_arch()

    with TemporaryDirectory() as root_dir:
        path = os.path.join(root_dir, 'kern')
        cu_path = '%s.cu' % path

        with open(cu_path, 'w') as cu_file:
            cu_file.write(source)

        prog = Program(six.b(source), six.b(os.path.basename(cu_path)))
        ptx = prog.compile([six.b('-arch={}'.format(arch))])

        return six.b(ptx)


def preprocess(source, options=()):
    pp_src = Program(six.b(source), six.b('')).compile()
    if isinstance(pp_src, six.binary_type):
        pp_src = pp_src.decode('utf-8')
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

    # if 'win32' == sys.platform:
    #     options += ('-Xcompiler', '/wd 4819')
    #     if sys.maxsize == 9223372036854775807:
    #         options += '-m64',
    #     elif sys.maxsize == 2147483647:
    #         options += '-m32',

    env = (arch, options, _get_nvrtc_version())
    if '#include' in source:
        pp_src = '%s %s' % (env, preprocess(source, options))
    else:
        base = _empty_file_preprocess_cache.get(env, None)
        if base is None:
            base = _empty_file_preprocess_cache[env] = preprocess('', options)
        pp_src = '%s %s %s' % (env, base, source)

    if isinstance(pp_src, six.text_type):
        pp_src = pp_src.encode('utf-8')
    name = '%s.cubin' % hashlib.md5(pp_src).hexdigest()

    mod = function.Module()

    if not os.path.isdir(cache_dir):
        try:
            os.makedirs(cache_dir)
        except OSError:
            if not os.path.isdir(cache_dir):
                raise

    lock_path = os.path.join(cache_dir, 'lock_file.lock')

    path = os.path.join(cache_dir, name)
    with filelock.FileLock(lock_path) as lock:
        if os.path.exists(path):
            with open(path, 'rb') as file:
                cubin = file.read()
        else:
            lock.release()
            cubin = nvrtc(source, options, arch)
            lock.acquire()
            with open(path, 'wb') as cubin_file:
                cubin_file.write(cubin)

    mod.load(cubin)

    return mod
