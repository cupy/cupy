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
import threading

from cupy import __version__ as cupy_ver
from cupy import _util
from cupy._environment import (get_nvcc_path, get_cuda_path)
from cupy.core import ndarray
from cupy.cuda.compiler import (_get_bool_env_variable, CompileException)
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


_callback_mgr = []
_callback_dev_code = None
_callback_thread_local = threading.local()
_default_cache_dir = os.path.expanduser('~/.cupy/callback_cache')


def _get_cache_dir():
    return os.environ.get('CUPY_CACHE_DIR', _default_cache_dir)


def get_current_callback_manager():
    if not hasattr(_callback_thread_local, '_current_cufft_callback'):
        _callback_thread_local._current_cufft_callback = None
    return _callback_thread_local._current_cufft_callback


# TODO(leofang): find a way to implement a lock-free method for
# cached shared libraries like what's done in cupy/cuda/compiler.py
class _CallbackManager:
    def __init__(
            self, cb_load='', cb_store='', cb_load_aux_arr=None,
            cb_store_aux_arr=None):
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
        if cb_load_aux_arr is not None:
            if not isinstance(cb_load_aux_arr, ndarray):
                raise ValueError('cb_load_aux_arr must be a cupy.ndarray')
            if not cb_load:
                raise ValueError('load callback is not given')
        if cb_store_aux_arr is not None:
            if not isinstance(cb_store_aux_arr, ndarray):
                raise ValueError('cb_store_aux_arr must be a cupy.ndarray')
            if not cb_store:
                raise ValueError('store callback is not given')

        self.cb_load = cb_load
        self.cb_store = cb_store
        self.cb_load_aux_arr = cb_load_aux_arr
        self.cb_store_aux_arr = cb_store_aux_arr

        # Set up some variables...
        cc = sysconfig.get_config_var('CXX').split(' ')
        python_include = sysconfig.get_path('include')
        cuda_include = get_cuda_path() + '/include/'
        cupy_root = os.path.join(os.path.dirname(__file__), '..')
        cupy_include = cupy_root + '/core/include'
        arch = get_compute_capability()
        build_ver = get_build_version()
        cufft_ver = get_cufft_version()
        source_dir = cupy_root + '/cuda/'
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

        # For hash; note this is independent of the plan to be created
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
                               '-I' + cuda_include,
                               '-I' + cupy_include,
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
                   '-I' + cupy_include,
                   '-c', self.dir + '/cupy_cufftXt.cu',
                   '-Xcompiler', '-fPIC', '-O2', '-std=c++11']
            if self.cb_load:
                cmd.append('-DHAS_LOAD_CALLBACK')
            if self.cb_store:
                cmd.append('-DHAS_STORE_CALLBACK')
            p = subprocess.run(cmd + ['-o', self.obj_dev],
                               env=os.environ,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
            try:
                p.check_returncode()
            except subprocess.CalledProcessError as e:
                cex = CompileException(
                    str(e) + '\nStderr: ' + e.stderr.decode(), support,
                    self.dir + '/cupy_cufftXt.cu',
                    cmd[1:], 'nvcc')
                dump = _get_bool_env_variable(
                    'CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
                if dump:
                    cex.dump(sys.stderr)
                raise cex

            # Use nvcc to link and generate a shared library, and place it in
            # the disk cache
            # WARNING: CANNOT use host compiler to link!
            p = subprocess.run([nvcc, '-ccbin', cc[0],
                                '-shared', '-arch=sm_'+arch,
                                self.obj_dev, self.obj_host,
                                '-lcufft_static', '-lculibos',
                                '-o', path],
                               env=os.environ)
            p.check_returncode()

            # Clean up build directory
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

    def create_plan(self, plan_info):
        plan_type, plan_args = plan_info
        plan = getattr(self.mod, plan_type)(*plan_args)
        self.plan = plan  # retain the most recently used plan
        return plan

    def set_callbacks(
            self, plan=None, cb_load_aux_arr=None, cb_store_aux_arr=None):
        if plan is None:
            # TODO(leofang): raise warning?
            plan = self.plan
        if cb_load_aux_arr is None:
            cb_load_aux_arr = self.cb_load_aux_arr
        if cb_store_aux_arr is None:
            cb_store_aux_arr = self.cb_store_aux_arr

        fft_type = plan.fft_type
        if fft_type == CUFFT_C2C:
            cb_load_type = CUFFT_CB_LD_COMPLEX if self.cb_load else -1
            cb_store_type = CUFFT_CB_ST_COMPLEX if self.cb_store else -1
        elif fft_type == CUFFT_R2C:
            cb_load_type = CUFFT_CB_LD_REAL if self.cb_load else -1
            cb_store_type = CUFFT_CB_ST_COMPLEX if self.cb_store else -1
        elif fft_type == CUFFT_C2R:
            cb_load_type = CUFFT_CB_LD_COMPLEX if self.cb_load else -1
            cb_store_type = CUFFT_CB_ST_REAL if self.cb_store else -1
        elif fft_type == CUFFT_Z2Z:
            cb_load_type = CUFFT_CB_LD_COMPLEX_DOUBLE if self.cb_load else -1
            cb_store_type = CUFFT_CB_ST_COMPLEX_DOUBLE if self.cb_store else -1
        elif fft_type == CUFFT_D2Z:
            cb_load_type = CUFFT_CB_LD_REAL_DOUBLE if self.cb_load else -1
            cb_store_type = CUFFT_CB_ST_COMPLEX_DOUBLE if self.cb_store else -1
        elif fft_type == CUFFT_Z2D:
            cb_load_type = CUFFT_CB_LD_COMPLEX_DOUBLE if self.cb_load else -1
            cb_store_type = CUFFT_CB_ST_REAL_DOUBLE if self.cb_store else -1
        else:
            raise ValueError

        if self.cb_load:
            if cb_load_aux_arr is not None:
                cb_load_ptr = cb_load_aux_arr.data.ptr
            else:
                cb_load_ptr = 0
            self.mod.setCallback(
                plan.handle, cb_load_type, True, cb_load_ptr)
        if self.cb_store:
            if cb_store_aux_arr is not None:
                cb_store_ptr = cb_store_aux_arr.data.ptr
            else:
                cb_store_ptr = 0
            self.mod.setCallback(
                plan.handle, cb_store_type, False, cb_store_ptr)


@contextlib.contextmanager
def set_cufft_callbacks(
        cb_load='', cb_store='', *,
        cb_load_aux_arr=None, cb_store_aux_arr=None):
    """A context manager for setting up load and/or store callbacks.

    Args:
        cb_load (str): A string contains the device kernel for the load
            callback. It must define ``d_loadCallbackPtr``.
        cb_store (str): A string contains the device kernel for the store
            callback. It must define ``d_storeCallbackPtr``.
        cb_load_aux_arr (:class:`cupy.ndarray`, optional): A CuPy array
            containing data to be used in the load callback.
        cb_store_aux_arr (:class:`cupy.ndarray`, optional): A CuPy array
            containing data to be used in the store callback.

    Yields:
        :class:`_CallbackManager`: A manager object handling the callbacks.
            This instance should not be used by users, except when the
            auxiliary arrays need to be updated.

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
    try:
        mgr = _CallbackManager(
            cb_load=cb_load,
            cb_store=cb_store,
            cb_load_aux_arr=cb_load_aux_arr,
            cb_store_aux_arr=cb_store_aux_arr)
        _callback_thread_local._current_cufft_callback = mgr
        _callback_mgr.append(mgr)  # keep the manager alive
        yield mgr
    finally:
        _callback_thread_local._current_cufft_callback = None
        # do not remove mgr from _callback_mgr, as one might still wanna use
        # it in the same Python session
