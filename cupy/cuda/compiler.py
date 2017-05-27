import hashlib
import os
import re
import shutil
import tempfile

from pynvrtc import compiler
import six

from cupy.cuda import device
from cupy.cuda import function

_nvrtc_version = None


def _get_nvrtc_version():
    global _nvrtc_version
    if _nvrtc_version is None:
        interface = compiler.NVRTCInterface()
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

    options += ('-arch={}'.format(arch),)

    with TemporaryDirectory() as root_dir:
        path = os.path.join(root_dir, 'kern')
        cu_path = '%s.cu' % path

        with open(cu_path, 'w') as cu_file:
            cu_file.write(source)

        prog = compiler.Program(
            six.b(source), six.b(os.path.basename(cu_path)))
        ptx = prog.compile([six.b(o) for o in options])

        return six.b(ptx)


def preprocess(source, options=()):
    pp_src = compiler.Program(six.b(source), six.b('')).compile()
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

    options += ('-ftz=true',)

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

    ptx = nvrtc(source, options, arch)
    if isinstance(ptx, six.text_type):
        ptx = ptx.encode('utf-8')
    ptx = six.b(ptx)
    ls = function.LinkState()
    ls.add_ptr_data(ptx, 'cupy.ptx')
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

    mod.load(cubin)
    return mod
