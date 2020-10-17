import contextlib
import hashlib
import importlib
import os
import shutil
import string
import subprocess
import sys
import sysconfig
import tempfile

from cupy import __version__ as cupy_ver
from cupy import _util
from cupy._environment import get_nvcc_path
from cupy.cuda.cufft import (CUFFT_C2C, CUFFT_C2R, CUFFT_R2C,
                             CUFFT_Z2Z, CUFFT_Z2D, CUFFT_D2Z,
                             CUFFT_CB_LD_COMPLEX, CUFFT_CB_LD_COMPLEX_DOUBLE,
                             CUFFT_CB_LD_REAL, CUFFT_CB_LD_REAL_DOUBLE,
                             CUFFT_CB_ST_COMPLEX, CUFFT_CB_ST_COMPLEX_DOUBLE,
                             CUFFT_CB_ST_REAL, CUFFT_CB_ST_REAL_DOUBLE,)
from cupy.cuda.cufft import getVersion as get_cufft_version
from cupy.cuda.device import get_compute_capability
from cupy_backends.cuda.api.driver import get_build_version
from cupy_backends.cuda.api.runtime import is_hip

# expose cache handles to this module
from cupy.fft._cache import get_plan_cache  # NOQA
from cupy.fft._cache import clear_plan_cache  # NOQA
from cupy.fft._cache import get_plan_cache_size  # NOQA
from cupy.fft._cache import set_plan_cache_size  # NOQA
from cupy.fft._cache import get_plan_cache_max_memsize  # NOQA
from cupy.fft._cache import set_plan_cache_max_memsize  # NOQA
from cupy.fft._cache import show_plan_cache_info  # NOQA


enable_nd_planning = True
use_multi_gpus = False
_devices = None


def set_cufft_gpus(gpus):
    '''Set the GPUs to be used in multi-GPU FFT.

    Args:
        gpus (int or list of int): The number of GPUs or a list of GPUs
            to be used. For the former case, the first ``gpus`` GPUs
            will be used.

    .. warning::
        This API is currently experimental and may be changed in the future
        version.

    .. seealso:: `Multiple GPU cuFFT Transforms`_

    .. _Multiple GPU cuFFT Transforms:
        https://docs.nvidia.com/cuda/cufft/index.html#multiple-GPU-cufft-transforms
    '''
    _util.experimental('cupy.fft.config.set_cufft_gpus')
    global _devices

    if isinstance(gpus, int):
        devs = [i for i in range(gpus)]
    elif isinstance(gpus, list):
        devs = gpus
    else:
        raise ValueError("gpus must be an int or a list of int.")
    if len(devs) <= 1:
        raise ValueError("Must use at least 2 GPUs.")

    # make it hashable
    _devices = tuple(devs)


_callback_load = ''
_callback_store = ''
_callback_mgr = []
_callback_dev_code = None
_default_cache_dir = os.path.expanduser('~/.cupy/callback_cache')


def _get_cache_dir():
    return os.environ.get('CUPY_CACHE_DIR', _default_cache_dir)


# TODO(leofang): would it be more robust if we use distutils +
# setuptools here? I'm worried that the number of lines of code
# might inflate too much...
# TODO(leofang): find a way to implement a lock-free method for
# cached shared libraries like what's done in cupy/cuda/compiler.py
# TODO(leofang): investigate if callerInfo can be supported. Looks
# like in that case we can't cache the Python modules?
class _CallbackManager:
    def __init__(self, plan_args, cb_load='', cb_store=''):
        # Sanity checks
        if is_hip:
            raise RuntimeError('hipFFT does not support callbacks')
        if not sys.platform.startswith('linux'):
            raise RuntimeError('cuFFT callbacks are only available on Linux')
        if not (sys.maxsize > 2**32):
            raise RuntimeError('cuFFT callbacks require 64 bit OS')
        if not cb_load and not cb_store:
            raise ValueError('need to specify either cb_load or cb_store, '
                             'or both')
        if cb_load and 'd_loadCallbackPtr' not in cb_load:
            raise ValueError('need to specify d_loadCallbackPtr in cb_load')
        if cb_store and 'd_storeCallbackPtr' not in cb_store:
            raise ValueError('need to specify d_storeCallbackPtr in cb_store')
        nvcc = get_nvcc_path()
        if nvcc is None:
            raise RuntimeError('nvcc is required but not found')
        try:
            import cython
        except ImportError:
            raise RuntimeError('cython is required but not found')
        else:
            del cython
        self.plan_args = plan_args[1]
        self.cb_load = cb_load
        self.cb_store = cb_store

        # Set up some variables...
        cc = sysconfig.get_config_var('CXX').split(' ')
        python_include = sysconfig.get_path('include')
        arch = get_compute_capability()
        build_ver = get_build_version()
        cufft_ver = get_cufft_version()
        source_dir = os.path.dirname(__file__) + '/../cuda/'
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

        # For hash; note this is independent of plan args
        keys = (cc, arch, build_ver, cufft_ver, ext_suffix, cupy_ver,
                cb_load, cb_store)
        keys = '%s %s %s %s %s %s %s %s' % keys

        # Generate module filename: all modules with the identical callbacks
        # are considered identical regardless of which plan was actually
        # executed at the time of generation
        mod_name = 'cupy_callback_'
        mod_name += hashlib.md5(keys.encode()).hexdigest()
        mod_name = mod_name.replace('.', '')
        mod_filename = mod_name + ext_suffix

        # Check if the module is already cached on disk. If not, we compile.
        cache_dir = _get_cache_dir() + '/'
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, mod_filename)
        if not os.path.isfile(path):
            # Set up temp directory
            self.dir_obj = tempfile.TemporaryDirectory()
            self.dir = self.dir_obj.name + '/'

            # Cythonize the Cython code to produce a c++ source file
            shutil.copyfile(source_dir + '/cupy_cufft.h',
                            self.dir + '/cupy_cufft.h')
            shutil.copyfile(source_dir + '/cufft.pxd',
                            self.dir + mod_name + '.pxd')
            shutil.copyfile(source_dir + '/cufft.pyx',
                            self.dir + mod_name + '.pyx')
            p = subprocess.run(['cython', '-3', '--cplus',
                                '-E', 'use_hip=0',
                                '-E', 'CUDA_VERSION='+str(build_ver),
                                '-E', 'CUPY_CUFFT_STATIC=True',
                                self.dir + mod_name + '.pyx',
                                '-o', self.dir + mod_name + '.cpp'],
                               env=os.environ)
            p.check_returncode()

            # Compile the Python module
            self.obj_host = self.dir + mod_name + '.o'
            shutil.copyfile(source_dir + '/cupy_cufftXt.h',
                            self.dir + '/cupy_cufftXt.h')
            p = subprocess.run(cc + [
                               '-I' + python_include,
                               '-fPIC', '-O2', '-std=c++11',
                               '-c', self.dir + mod_name + '.cpp',
                               '-o', self.obj_host],
                               env=os.environ)
            p.check_returncode()

            # Dump and compile device code using nvcc
            global _callback_dev_code
            if _callback_dev_code is None:
                with open(source_dir + '/cupy_cufftXt.cu') as f:
                    support = _callback_dev_code = f.read()
            else:
                support = _callback_dev_code
            with open(self.dir + '/cupy_cufftXt.cu', 'w') as f:
                support = string.Template(support).substitute(
                    dev_load_callback_ker=cb_load,
                    dev_store_callback_ker=cb_store)
                f.write(support)
            self.obj_dev = self.dir + mod_name + '_dev.o'
            cmd = [nvcc, '-ccbin', cc[0],
                   '-arch=sm_'+arch, '-dc',
                   '-c', self.dir + '/cupy_cufftXt.cu',
                   '-Xcompiler', '-fPIC', '-O2', '-std=c++11']
            if self.cb_load:
                cmd.append('-DHAS_LOAD_CALLBACK')
            if self.cb_store:
                cmd.append('-DHAS_STORE_CALLBACK')
            p = subprocess.run(cmd + ['-o', self.obj_dev], env=os.environ)
            p.check_returncode()

            # Use nvcc to link and generate a shared library
            # WARNING: CANNOT use host compiler to link!
            p = subprocess.run([nvcc, '-ccbin', cc[0],
                                '-shared', '-arch=sm_'+arch,
                                self.obj_dev, self.obj_host,
                                '-lcufft_static', '-lculibos',
                                '-o', path],
                               env=os.environ)
            p.check_returncode()

            self.dir_obj.cleanup()
            del self.dir_obj
            del self.dir
            del self.obj_host
            del self.obj_dev

        # Load the Python module
        spec = importlib.util.spec_from_file_location(mod_name, path)
        self.mod = module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

        # Create a cuFFT plan
        plan_type = plan_args[0]
        self.plan = getattr(self.mod, plan_type)(*plan_args[1])
        self.handle = self.plan.handle
        self.fft_type = self.plan.fft_type
        self.is_callback_set = False

    def set_callbacks(self, cb_load_type=-1, cb_store_type=-1):
        if self.is_callback_set:
            raise RuntimeError('callback cannot be reset')
        if self.cb_load:
            if cb_load_type == -1:
                raise ValueError('cb_load_type needs to be speficied')
            self.mod.setCallback(self.handle, cb_load_type, True)
        if self.cb_store:
            if cb_store_type == -1:
                raise ValueError('cb_store_type needs to be speficied')
            self.mod.setCallback(self.handle, cb_store_type, False)
        self.is_callback_set = True


@contextlib.contextmanager
def set_cufft_callbacks(cb_load='', cb_store=''):
    """A context manager for setting up load and/or store callbacks.

    Args:
        cb_load (str): A string contains the device kernel for the load
            callback. It must define ``d_loadCallbackPtr``.
        cb_store (str): A string contains the device kernel for the store
            callback. It must define ``d_storeCallbackPtr``.

    .. note::
        Any FFT calls living in this context will have callbacks set up. An
        example for a load callback is shown below:

        .. code-block:: python

            code = r'''
            __device__ cufftComplex CB_ConvertInputC(
                void *dataIn,
                size_t offset,
                void *callerInfo,
                void *sharedPtr) {
              // implementation
            }

            __device__ cufftCallbackLoadC d_loadCallbackPtr = CB_ConvertInputC;
            '''

            with cp.fft.config.set_cufft_callbacks(cb_load=code):
                out_arr = cp.fft.fft(in_arr, ...)

    .. warning::
        Using cuFFT callbacks requires compiling and loading a Python module at
        runtime as well as static linking for each distinct transform and
        callback, so the first invocation for each combination will be very
        slow. This is a limitation of cuFFT, so use this feature only when the
        callback-enabled transform is known more performant and can be reused
        to amortize the cost.

    .. warning::
        The generated Python modules are by default cached in
        ``~/.cupy/callback_cache`` for possible reuse (with the same set of
        load/store callbacks). Due to static linking, however, the file sizes
        can be excessive! The cache position can be changed via setting
        ``CUPY_CACHE_DIR``.

    """
    global _callback_load, _callback_store
    try:
        if cb_load:
            _callback_load = cb_load
        if cb_store:
            _callback_store = cb_store
        yield
    finally:
        _callback_load = ''
        _callback_store = ''


def _get_static_plan(plan_type, fft_type, keys):
    global _callback_load, _callback_store, _callback_mgr

    mgr = _CallbackManager(
        (plan_type, keys), cb_load=_callback_load, cb_store=_callback_store)
    if fft_type == CUFFT_C2C:
        cb_load_type = CUFFT_CB_LD_COMPLEX if _callback_load else -1
        cb_store_type = CUFFT_CB_ST_COMPLEX if _callback_store else -1
    elif fft_type == CUFFT_R2C:
        cb_load_type = CUFFT_CB_LD_REAL if _callback_load else -1
        cb_store_type = CUFFT_CB_ST_COMPLEX if _callback_store else -1
    elif fft_type == CUFFT_C2R:
        cb_load_type = CUFFT_CB_LD_COMPLEX if _callback_load else -1
        cb_store_type = CUFFT_CB_ST_REAL if _callback_store else -1
    elif fft_type == CUFFT_Z2Z:
        cb_load_type = CUFFT_CB_LD_COMPLEX_DOUBLE if _callback_load else -1
        cb_store_type = CUFFT_CB_ST_COMPLEX_DOUBLE if _callback_store else -1
    elif fft_type == CUFFT_D2Z:
        cb_load_type = CUFFT_CB_LD_REAL_DOUBLE if _callback_load else -1
        cb_store_type = CUFFT_CB_ST_COMPLEX_DOUBLE if _callback_store else -1
    elif fft_type == CUFFT_Z2D:
        cb_load_type = CUFFT_CB_LD_COMPLEX_DOUBLE if _callback_load else -1
        cb_store_type = CUFFT_CB_ST_REAL_DOUBLE if _callback_store else -1
    else:
        raise ValueError
    mgr.set_callbacks(cb_load_type, cb_store_type)

    _callback_load = ''
    _callback_store = ''
    _callback_mgr.append(mgr)  # keep the manager alive
    return mgr.plan
