import contextlib

from cupy import _util
from cupy.cuda.cufft import (CUFFT_C2C, CUFFT_C2R, CUFFT_R2C,
                             CUFFT_Z2Z, CUFFT_Z2D, CUFFT_D2Z,
                             CUFFT_CB_LD_COMPLEX, CUFFT_CB_LD_COMPLEX_DOUBLE,
                             CUFFT_CB_LD_REAL, CUFFT_CB_LD_REAL_DOUBLE,
                             CUFFT_CB_ST_COMPLEX, CUFFT_CB_ST_COMPLEX_DOUBLE,
                             CUFFT_CB_ST_REAL, CUFFT_CB_ST_REAL_DOUBLE,)
from cupy.cuda.device import get_compute_capability
from cupy_backends.cuda.api.driver import get_build_version

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


class _CallbackManager:
    def __init__(self, plan_args, cb_load='', cb_store=''):
        import importlib
        import os
        import shutil
        import string
        import subprocess
        import sys
        import sysconfig
        import tempfile
        from cupy._environment import get_nvcc_path

        # TODO(leofang): cache the generated .so file based on:
        #   - Python version
        #   - CuPy version
        #   - cuFFT version
        #   - plan_args
        #   - cb_load & cb_store

        # TODO(leofang): would it be more robust if we use distutils +
        # setuptools here? I'm worried that the number of lines of code
        # might inflate too much...

        # TODO(leofang): make sure Cython is installed at runtime

        # TODO(leofang): make sure all needed source files are included
        # in sdist/wheel

        if not cb_load and not cb_store:
            raise ValueError('need to specify either cb_load or cb_store, '
                             'or both')
        if cb_load and 'd_loadCallbackPtr' not in cb_load:
            raise ValueError('need to specify d_loadCallbackPtr in cb_load')
        if cb_store and 'd_storeCallbackPtr' not in cb_store:
            raise ValueError('need to specify d_storeCallbackPtr in cb_store')
        self.plan_args = plan_args[1]
        self.cb_load = cb_load
        self.cb_store = cb_store

        python_include = sysconfig.get_paths()['include']
        nvcc = get_nvcc_path()
        CC = get_compute_capability()
        build_ver = get_build_version()
        source_dir = os.path.dirname(__file__) + '/../cuda/'

        # Set up temp directory; its lifetime is tied with the present instance
        self.dir_obj = tempfile.TemporaryDirectory()
        self.dir = self.dir_obj.name

        # Cythonize the Cython code; a c++ source cupy_callback.cpp is produced
        shutil.copyfile(source_dir+'/cupy_cufft.h', self.dir+'/cupy_cufft.h')
        shutil.copyfile(source_dir+'/cufft.pxd', self.dir+'/cupy_callback.pxd')
        shutil.copyfile(source_dir+'/cufft.pyx', self.dir+'/cupy_callback.pyx')
        p = subprocess.run(['cython', '-3', '--cplus',
                            '-E', 'use_hip=0',
                            '-E', 'CUDA_VERSION='+str(build_ver),
                            '-E', 'CUPY_CUFFT_STATIC=True',
                            self.dir+'/cupy_callback.pyx'],
                           env=os.environ)
        p.check_returncode()

        # Compile the Python module
        self.obj_host = self.dir+'/cupy_callback.o'
        shutil.copyfile(source_dir+'/cupy_cufftx.h', self.dir+'/cupy_cufftx.h')
        p = subprocess.run(['g++', '-I'+python_include,
                            '-fPIC', '-pthread', '-O2', '-std=c++11',
                            '-c', self.dir+'/cupy_callback.cpp',
                            '-o', self.obj_host],
                           env=os.environ)
        p.check_returncode()

        # Dump and compile device code using nvcc
        global _callback_dev_code
        if _callback_dev_code is None:
            with open(source_dir+'/cupy_cufftx.cu') as f:
                support = _callback_dev_code = f.read()
        else:
            support = _callback_dev_code
        with open(self.dir+'/cupy_cufftx.cu', 'w') as f:
            support = string.Template(support).substitute(
                dev_load_callback_ker=cb_load,
                dev_store_callback_ker=cb_store,
            )
            f.write(support)
        self.obj_dev = self.dir + '/cupy_callback_dev.o'
        cmd = [nvcc, '-ccbin', 'g++', '-arch=sm_'+CC, '-dc',
               '-c', self.dir+'/cupy_cufftx.cu',
               '-Xcompiler', '-fPIC', '-O2', '-std=c++11']
        if self.cb_load:
            cmd.append('-DHAS_LOAD_CALLBACK')
        if self.cb_store:
            cmd.append('-DHAS_STORE_CALLBACK')
        p = subprocess.run(cmd + ['-o', self.obj_dev], env=os.environ)
        p.check_returncode()

        # Use nvcc to link and generate a shared library
        # WARNING: CANNOT use host compiler to link!
        self.lib = self.dir + '/cupy_callback.so'
        p = subprocess.run([nvcc, '-ccbin', 'g++', '-shared', '-arch=sm_'+CC,
                            self.obj_dev, self.obj_host,
                            '-lcufft_static', '-lculibos', '-o', self.lib],
                           env=os.environ)
        p.check_returncode()

        # Load the Python module
        spec = importlib.util.spec_from_file_location(
            'cupy_callback', self.lib)
        self.mod = module = importlib.util.module_from_spec(spec)
        sys.modules['cupy_callback'] = module
        spec.loader.exec_module(module)

        # Create a cuFFT plan
        plan = plan_args[0]
        self.plan = getattr(self.mod, plan)(*plan_args[1])
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

    def __del__(self):
        self.dir_obj.cleanup()


@contextlib.contextmanager
def set_callbacks(cb_load='', cb_store=''):
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

            with cp.fft.config.set_callbacks(cb_load=code):
                out_arr = cp.fft.fft(in_arr, ...)

    .. warning::
        Using cuFFT callbacks requires compiling and loading an FFT module as
        well as static linking for each distinct transform or callback, so the
        first invocation would be slow. This is a limitation of cuFFT, so use
        this feature only when the callback-enabled transform can be reused
        to amortize the cost.

    .. warning::
        When a statically-linked callback-enabled plan is generated (likely
        cached), the callbacks set by this function will be reset. To enable
        callbacks for the next transform, call this function again.

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
