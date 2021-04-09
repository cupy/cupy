from libc.stdint cimport intptr_t

from cupy_backends.cuda.api.driver cimport get_build_version
from cupy_backends.cuda.api.runtime cimport _is_hip_environment
from cupy._core.core cimport ndarray
from cupy.cuda.device cimport get_compute_capability

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
import warnings

from cupy import __version__ as _cupy_ver
from cupy._environment import (get_nvcc_path, get_cuda_path)
from cupy.cuda.compiler import (_get_bool_env_variable, CompileException)
# for some reason we can't cimport stuff from cupy.cuda.cufft...
from cupy.cuda.cufft import (CUFFT_C2C, CUFFT_C2R, CUFFT_R2C,
                             CUFFT_Z2Z, CUFFT_Z2D, CUFFT_D2Z,
                             CUFFT_CB_LD_COMPLEX, CUFFT_CB_LD_COMPLEX_DOUBLE,
                             CUFFT_CB_LD_REAL, CUFFT_CB_LD_REAL_DOUBLE,
                             CUFFT_CB_ST_COMPLEX, CUFFT_CB_ST_COMPLEX_DOUBLE,
                             CUFFT_CB_ST_REAL, CUFFT_CB_ST_REAL_DOUBLE,)
from cupy.cuda.cufft import getVersion as get_cufft_version


# information needed for building an external module
cdef list _cc = []
cdef str _python_include = None
cdef list _nvcc = []
cdef str _cuda_path = None
cdef str _cuda_include = None
cdef str _nvprune = None
cdef str _build_ver = None
cdef int _cufft_ver = 0
cdef str _cupy_root = None
cdef str _cupy_include = None
cdef str _source_dir = None
cdef str _ext_suffix = None


# callback related stuff
cdef bint _is_init = False
cdef str _callback_dev_code = None
cdef str _callback_cache_dir = None
cdef dict _callback_mgr = {}  # keep the Python modules alive
cdef object _callback_thread_local = threading.local()


cdef class _ThreadLocal:
    cdef _CallbackManager _current_cufft_callback

    def __init__(self):
        self._current_cufft_callback = None

    @staticmethod
    cdef _ThreadLocal get():
        cdef _ThreadLocal tls
        try:
            tls = _callback_thread_local.tls
        except AttributeError:
            tls = _callback_thread_local.tls = _ThreadLocal()
        return tls


cdef inline void _set_vars() except*:
    global _cc, _python_include, _cuda_path, _cuda_include, _nvprune
    global _build_ver, _cufft_ver, _ext_suffix, _is_init

    cdef str cxx = sysconfig.get_config_var('CXX')
    if cxx is not None:
        _cc = cxx.split(' ')
    elif 'CXX' in os.environ:
        _cc = os.environ.get('CXX').split(' ')
    else:
        _cc = None

    _python_include = sysconfig.get_path('include')

    _cuda_path = get_cuda_path()
    if _cuda_path is not None:
        _cuda_include = os.path.join(_cuda_path, 'include')
        _nvprune = os.path.join(_cuda_path, 'bin/nvprune')
        if not os.path.isfile(_nvprune):
            _nvprune = None

    _build_ver = str(get_build_version())
    _cufft_ver = get_cufft_version()
    _ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

    _set_cupy_paths()
    _set_nvcc_path()
    _is_init = True


cdef inline void _set_cupy_paths() except*:
    # Workaround: older Cython cannot use __file__ in global scope
    global _cupy_root, _cupy_include, _source_dir
    if _cupy_root is None:
        _cupy_root = os.path.join(os.path.dirname(__file__), '..')
        _cupy_include = os.path.join(_cupy_root, '_core/include')
        _source_dir = os.path.join(_cupy_root, 'cuda')


cdef inline void _set_nvcc_path() except*:
    # get_nvcc_path() could be a long string like "ccache nvcc ..." from NVCC
    cdef str nvcc = None
    cdef str f
    cdef int i

    global _nvcc
    if _nvcc == []:
        nvcc = get_nvcc_path()
        if nvcc is not None:
            _nvcc = nvcc.split(' ')
            # if the host compiler is set in NVCC, we force it to align with
            # the one reported by sysconfig
            for i, f in enumerate(_nvcc):
                if f in ('-ccbin', '--compiler-bindir'):
                    _nvcc[i+1] = _cc[0]
                    break
            else:
                _nvcc += ['-ccbin', _cc[0]]
        else:
            _nvcc = None


cdef inline void _sanity_checks(
        str cb_load, str cb_store,
        ndarray cb_load_aux_arr, ndarray cb_store_aux_arr) except*:
    if _is_hip_environment:
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
    if _cc is None:
        raise RuntimeError('a C++ compiler is required but not found, '
                           'please set the environment variable CXX')
    if _nvcc is None:
        raise RuntimeError('nvcc is required but not found')
    if _nvprune is None:
        warnings.warn('nvprune is not found', RuntimeWarning)
    if cb_load_aux_arr is not None:
        if not cb_load:
            raise ValueError('load callback is not given')
    if cb_store_aux_arr is not None:
        if not cb_store:
            raise ValueError('store callback is not given')


cdef inline void _cythonize(str tempdir, str mod_name) except*:
    shutil.copyfile(os.path.join(_source_dir, 'cupy_cufft.h'),
                    os.path.join(tempdir, 'cupy_cufft.h'))
    shutil.copyfile(os.path.join(_source_dir, 'cufft.pxd'),
                    os.path.join(tempdir, mod_name + '.pxd'))
    shutil.copyfile(os.path.join(_source_dir, 'cufft.pyx'),
                    os.path.join(tempdir, mod_name + '.pyx'))
    p = subprocess.run(['cython', '-3', '--cplus',
                        '-E', 'HIP_VERSION=0',
                        '-E', 'CUDA_VERSION=' + _build_ver,
                        '-E', 'CUPY_CUFFT_STATIC=True',
                        os.path.join(tempdir, mod_name + '.pyx'),
                        '-o', os.path.join(tempdir, mod_name + '.cpp')],
                       env=os.environ, cwd=tempdir)
    p.check_returncode()


cdef inline void _mod_compile(str tempdir, str mod_name, str obj_host) except*:
    shutil.copyfile(os.path.join(_source_dir, 'cupy_cufftXt.h'),
                    os.path.join(tempdir, 'cupy_cufftXt.h'))
    p = subprocess.run(_cc + [
                       '-I' + _python_include,
                       '-I' + _cuda_include,
                       '-I' + _cupy_include,
                       '-fPIC', '-O2', '-std=c++11',
                       '-c', os.path.join(tempdir, mod_name + '.cpp'),
                       '-o', obj_host],
                       env=os.environ, cwd=tempdir)
    p.check_returncode()


cdef inline str _prune(str temp_dir, str cache_dir, str _cufft_ver, str arch):
    cdef str cufft_lib_full, cufft_lib_pruned, cufft_lib_temp, cufft_lib_cached

    if _nvprune:
        cufft_lib_full = os.path.join(_cuda_path, 'lib64/libcufft_static.a')
        cufft_lib_pruned = 'cufft_static_' + _cufft_ver + '_sm' + arch
        cufft_lib_temp = os.path.join(temp_dir,
                                      'lib' + cufft_lib_pruned + '.a')
        cufft_lib_cached = os.path.join(cache_dir,
                                        'lib' + cufft_lib_pruned + '.a')
        if not os.path.isfile(cufft_lib_cached):
            p = subprocess.run([_nvprune, '-arch=sm_' + arch,
                                cufft_lib_full, '-o', cufft_lib_temp],
                               env=os.environ, cwd=temp_dir)
            p.check_returncode()
            # atomic move with the destination guaranteed to be overwritten;
            # using os.replace() is also ok here
            os.rename(cufft_lib_temp, cufft_lib_cached)
    else:
        # nvprune is not found, just link against the full static lib
        cufft_lib_pruned = None

    return cufft_lib_pruned


cdef inline void _nvcc_compile(
        str tempdir, str mod_name, str cb_load, str cb_store,
        str obj_dev, str arch) except*:
    cdef str support
    cdef list cmd
    global _callback_dev_code

    if _callback_dev_code is None:
        with open(os.path.join(_source_dir, 'cupy_cufftXt.cu')) as f:
            support = _callback_dev_code = f.read()
    else:
        support = _callback_dev_code
    with open(os.path.join(tempdir, 'cupy_cufftXt.cu'), 'w') as f:
        support = string.Template(support).substitute(
            dev_load_callback_ker=cb_load,
            dev_store_callback_ker=cb_store)
        f.write(support)
    cmd = _nvcc + ['-arch=sm_'+arch, '-dc',
                   '-I' + _cupy_include,
                   '-c', os.path.join(tempdir, 'cupy_cufftXt.cu'),
                   '-Xcompiler', '-fPIC', '-O2', '-std=c++11']
    if cb_load:
        cmd.append('-DHAS_LOAD_CALLBACK')
    if cb_store:
        cmd.append('-DHAS_STORE_CALLBACK')
    p = subprocess.run(cmd + ['-o', obj_dev],
                       env=os.environ, cwd=tempdir,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    try:
        p.check_returncode()
    except subprocess.CalledProcessError as e:
        cex = CompileException(
            str(e) + '\nStderr: ' + e.stderr.decode(), support,
            os.path.join(tempdir, 'cupy_cufftXt.cu'),
            cmd[1:], 'nvcc')
        dump = _get_bool_env_variable(
            'CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
        if dump:
            cex.dump(sys.stderr)
        raise cex


cdef inline void _nvcc_link(
        str tempdir, str obj_host, str obj_dev, str arch, str mod_filename,
        str cache_dir, str cufft_lib_pruned) except*:
    # WARNING: CANNOT use host compiler to link!
    cdef list cmd
    cdef str mod_temp, mod_cached

    mod_temp = os.path.join(tempdir, mod_filename)
    mod_cached = os.path.join(cache_dir, mod_filename)
    cmd = _nvcc + ['-shared', '-arch=sm_'+arch, obj_dev, obj_host]
    if cufft_lib_pruned:
        cmd += ['-L'+cache_dir, '-l'+cufft_lib_pruned]
    else:
        cmd.append('-lcufft_static')
    cmd += ['-lculibos', '-lpthread', '-o', mod_temp]

    p = subprocess.run(cmd, env=os.environ, cwd=tempdir)
    p.check_returncode()
    # atomic move with the destination guaranteed to be overwritten;
    # using os.replace() is also ok here
    os.rename(mod_temp, mod_cached)


# make it a plain Python function so that we can mock-test it
def get_cache_dir():
    global _callback_cache_dir
    if _callback_cache_dir is None:
        _callback_cache_dir = os.environ.get(
            'CUPY_CACHE_DIR', os.path.expanduser('~/.cupy/callback_cache'))
    return _callback_cache_dir


cpdef get_current_callback_manager():
    cdef _ThreadLocal tls = _ThreadLocal.get()
    cdef _CallbackManager mgr = tls._current_cufft_callback
    return mgr


cdef class _CallbackManager:
    cdef:
        readonly str cb_load
        readonly str cb_store
        readonly ndarray cb_load_aux_arr
        readonly ndarray cb_store_aux_arr
        object mod

    def __init__(self,
                 str cb_load='',
                 str cb_store='',
                 ndarray cb_load_aux_arr=None,
                 ndarray cb_store_aux_arr=None):
        if not _is_init:
            _set_vars()
        _sanity_checks(cb_load, cb_store,
                       cb_load_aux_arr, cb_store_aux_arr)

        self.cb_load = cb_load
        self.cb_store = cb_store
        self.cb_load_aux_arr = cb_load_aux_arr
        self.cb_store_aux_arr = cb_store_aux_arr

        # Set up some variables...
        cdef str arch = get_compute_capability()
        cdef str tempdir
        cdef str obj_host
        cdef str obj_dev
        cdef str mod_name
        cdef str mod_filename
        cdef str cache_dir
        cdef str path
        cdef str cufft_lib_pruned

        # For hash; note this is independent of the plan to be created, and
        # only depends on the given load/store callbacks and the runtime
        # environment
        keys = (_cc, _nvcc, arch, _build_ver, _cufft_ver, _cupy_ver,
                _python_include, cb_load, cb_store)
        keys = '%s %s %s %s %s %s %s %s %s' % keys

        # Generate module filename: all modules with the identical callbacks
        # are considered identical regardless of which plan is actually
        # executed at the time of generation
        mod_name = 'cupy_callback_'
        mod_name += hashlib.md5(keys.encode()).hexdigest()
        mod_name = mod_name.replace('.', '')
        mod_filename = mod_name + _ext_suffix

        # Check if the module is already cached on disk. If not, we compile.
        cache_dir = get_cache_dir()
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, mod_filename)
        if not os.path.isfile(path):
            # Set up temp directory; it must be under the cache directory so
            # that atomic moves within the same filesystem can be guaranteed
            tempdir_obj = tempfile.TemporaryDirectory(dir=cache_dir)
            tempdir = tempdir_obj.name

            # Cythonize the Cython code to produce a c++ source file
            _cythonize(tempdir, mod_name)

            # Compile the Python module
            obj_host = os.path.join(tempdir, mod_name + '.o')
            _mod_compile(tempdir, mod_name, obj_host)

            # Prune libcufft_static.a for the target arch and cache it
            cufft_lib_pruned = _prune(
                tempdir, cache_dir, str(_cufft_ver), arch)

            # Dump and compile device code using nvcc
            obj_dev = os.path.join(tempdir, mod_name + '_dev.o')
            _nvcc_compile(tempdir, mod_name, cb_load, cb_store, obj_dev, arch)

            # Use nvcc to link and generate a shared library, and place it in
            # the disk cache
            _nvcc_link(tempdir, obj_host, obj_dev, arch, mod_filename,
                       cache_dir, cufft_lib_pruned)

            # Clean up build directory
            tempdir_obj.cleanup()

        # Load the Python module
        spec = importlib.util.spec_from_file_location(mod_name, path)
        self.mod = module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

    cpdef create_plan(self, tuple plan_info):
        cdef str plan_type
        cdef tuple plan_args

        plan_type, plan_args = plan_info
        plan = getattr(self.mod, plan_type)(*plan_args)
        return plan

    cpdef set_callbacks(self, plan):
        '''Set the load/store callbacks by making calls to
        ``cufftXtSetCallback``.

        Args:
            plan (:class:`~cupy.cuda.cufft.Plan1d` or
                :class:`~cupy.cuda.cufft.PlanNd`, optional): A cuFFT plan
                against which the load/store callbacks are set.

        .. note::
            If :meth:`set_caller_infos` is called, a call to this method must
            follow.

        '''
        cdef ndarray cb_load_aux_arr = self.cb_load_aux_arr
        cdef ndarray cb_store_aux_arr = self.cb_store_aux_arr
        cdef intptr_t cb_load_ptr=0, cb_store_ptr=0

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
            self.mod.setCallback(
                plan.handle, cb_load_type, True, cb_load_ptr)
        if self.cb_store:
            if cb_store_aux_arr is not None:
                cb_store_ptr = cb_store_aux_arr.data.ptr
            self.mod.setCallback(
                plan.handle, cb_store_type, False, cb_store_ptr)

    cdef set_caller_infos(self,
                          ndarray cb_load_aux_arr=None,
                          ndarray cb_store_aux_arr=None):
        '''Set the auxilliary arrays to be used by the load/store callbacks.
        Corresponding to the ``callerInfo`` field in cuFFT's callback API.

        Args:
            cb_load_aux_arr (:class:`cupy.ndarray`, optional): A CuPy array
                containing data to be used in the load callback.
            cb_store_aux_arr (:class:`cupy.ndarray`, optional): A CuPy array
                containing data to be used in the store callback.

        .. note::
            After this method is called, a call to :meth:`set_callbacks` must
            follow. This is for internal use.

        '''
        self.cb_load_aux_arr = cb_load_aux_arr
        self.cb_store_aux_arr = cb_store_aux_arr


cdef class set_cufft_callbacks:
    """A context manager for setting up load and/or store callbacks.

    Args:
        cb_load (str): A string contains the device kernel for the load
            callback. It must define ``d_loadCallbackPtr``.
        cb_store (str): A string contains the device kernel for the store
            callback. It must define ``d_storeCallbackPtr``.
        cb_load_aux_arr (cupy.ndarray, optional): A CuPy array containing
            data to be used in the load callback.
        cb_store_aux_arr (cupy.ndarray, optional): A CuPy array containing
            data to be used in the store callback.

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

    .. note::
        Below are the *runtime* requirements for using this feature:

            * cython >= 0.29.0
            * A host compiler that supports C++11 and above; might need to set
              up the ``CXX`` environment variable.
            * ``nvcc`` and the full CUDA Toolkit. Note that the ``cudatoolkit``
              package from Conda-Forge is not enough, as it does not contain
              static libraries.

    .. note::
        Callbacks only work for transforms over contiguous axes; the behavior
        for non-contiguous transforms is in general undefined.

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

    .. seealso:: `cuFFT Callback Routines`_

    .. _cuFFT Callback Routines:
        https://docs.nvidia.com/cuda/cufft/index.html#callback-routines

    """
    # this class should have been a simple function decorated by @contextlib.
    # contextmanager that yields...

    cdef:
        _CallbackManager mgr
        _CallbackManager mgr_prev

    def __init__(self,
                 str cb_load='',
                 str cb_store='',
                 *,
                 ndarray cb_load_aux_arr=None,
                 ndarray cb_store_aux_arr=None):
        # For every distinct pair of load & store callbacks, we compile an
        # external Python module and cache it.
        cdef tuple key = (cb_load, cb_store)
        cdef _CallbackManager mgr = _callback_mgr.get(key)
        if mgr is None:
            mgr = _CallbackManager(
                cb_load=cb_load,
                cb_store=cb_store,
                cb_load_aux_arr=cb_load_aux_arr,
                cb_store_aux_arr=cb_store_aux_arr)
            _callback_mgr[key] = mgr  # keep the Python module alive
        else:
            mgr.set_caller_infos(
                cb_load_aux_arr=cb_load_aux_arr,
                cb_store_aux_arr=cb_store_aux_arr)
        self.mgr = mgr

    def __enter__(self):
        cdef _ThreadLocal tls = _ThreadLocal.get()
        self.mgr_prev = tls._current_cufft_callback
        tls._current_cufft_callback = self.mgr
        return self.mgr

    def __exit__(self, exc_type, exc_value, traceback):
        cdef _ThreadLocal tls = _ThreadLocal.get()
        tls._current_cufft_callback = self.mgr_prev
        # do not remove mgr from _callback_mgr, as one might still wanna use
        # plans generated by it in the same Python session
