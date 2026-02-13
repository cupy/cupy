from __future__ import annotations

import inspect
import io
import os
import platform
import warnings

import numpy

import cupy
import cupy_backends


is_hip = cupy_backends.cuda.api.runtime.is_hip
if not is_hip:
    from cuda import pathfinder


def _eval_or_error(func, errors):
    # Evaluates `func` and return the result.
    # If an error specified by `errors` occurred, it returns a string
    # representing the error.
    try:
        return func()
    except errors as e:
        return repr(e)


def _load_and_get_path(lib_name):
    try:
        loaded_dl = pathfinder.load_nvidia_dynamic_lib(lib_name)
    except (pathfinder.DynamicLibNotFoundError, RuntimeError):
        return None
    else:
        return loaded_dl.abs_path


def _version_and_path(ver_seq, path_seq):
    assert len(ver_seq) >= len(path_seq)
    result = []
    for i in range(len(ver_seq)):
        result.append(ver_seq[i])
        if i < len(path_seq):
            result.append(path_seq[i])
    return result


class _InstallInfo:

    # TODO(niboshi): Add is_binary_distribution

    def __init__(self):
        cupy_package_root = self._get_cupy_package_root()
        if cupy_package_root is not None:
            data_root = os.path.join(cupy_package_root, '.data')
            data_paths = {
                'lib': _dir_or_none(os.path.join(data_root, 'lib')),
                'include': _dir_or_none(os.path.join(data_root, 'include')),
            }
        else:
            data_paths = {
                'lib': None,
                'include': None,
            }

        self.cupy_package_root = cupy_package_root
        self.data_paths = data_paths

    def get_data_path(self, data_type):
        if data_type not in self.data_paths:
            raise ValueError('Invalid data type: {}'.format(data_type))
        return self.data_paths[data_type]

    def _get_cupy_package_root(self):
        try:
            cupy_path = inspect.getfile(cupy)
        except TypeError:
            return None
        return os.path.dirname(cupy_path)


class _RuntimeInfo:

    cupy_version = None
    cuda_path = None
    nvcc_path = None

    # CUDA Driver
    cuda_build_version = None
    cuda_driver_version = None

    # CUDA Runtime
    cuda_runtime_version = None
    cuda_local_runtime_version = None
    cuda_local_runtime_path = None

    # CUDA Toolkit
    cublas_version = None
    cublas_path = None
    cufft_version = None
    cufft_path = None
    curand_version = None
    curand_path = None
    cusolver_version = None
    cusolver_path = None
    cusparse_version = None
    cusparse_path = None
    nvrtc_version = None
    nvrtc_path = None
    thrust_version = None
    cuda_extra_include_dirs = None

    # Optional Libraries
    nccl_build_version = None
    nccl_runtime_version = None
    nccl_path = None
    cub_build_version = None
    jitify_build_version = None
    cutensor_version = None
    cutensor_path = None
    cusparselt_version = None
    cusparselt_path = None
    cython_build_version = None
    cython_version = None

    numpy_version = None
    scipy_version = None

    _full = True

    def __init__(self, *, full=True):
        self._full = full
        self.cupy_version = cupy.__version__

        if not is_hip:
            self.cuda_path = cupy.cuda.get_cuda_path()
        else:
            self.cuda_path = cupy._environment.get_rocm_path()

        if not is_hip:
            self.nvcc_path = cupy._environment.get_nvcc_path()
        else:
            self.nvcc_path = cupy._environment.get_hipcc_path()

        # CUDA Driver
        self.cuda_build_version = str(cupy.cuda.driver.get_build_version())
        if cupy.cuda.driver._is_cuda_python():
            try:
                import cuda.bindings
                cuda_version = cuda.bindings.__version__
            except ImportError:
                import cuda
                cuda_version = cuda.__version__
            self.cuda_build_version += f' (CUDA Python: {cuda_version})'
        self.cuda_driver_version = _eval_or_error(
            cupy.cuda.runtime.driverGetVersion,
            cupy.cuda.runtime.CUDARuntimeError)

        # CUDA Runtime
        self.cuda_runtime_version = _eval_or_error(
            cupy.cuda.runtime.runtimeGetVersion,
            cupy.cuda.runtime.CUDARuntimeError)
        self.cuda_local_runtime_version = _eval_or_error(
            cupy.cuda.get_local_runtime_version,
            Exception)
        if full and not is_hip:
            self.cuda_local_runtime_path = _load_and_get_path('cudart')

        # cuBLAS
        self.cublas_version = '(available)'
        if full:
            self.cublas_version = _eval_or_error(
                lambda: cupy.cuda.cublas.getVersion(
                    cupy.cuda.device.get_cublas_handle()),
                Exception)
            if not is_hip:
                self.cublas_path = _load_and_get_path('cublas')

        # cuFFT
        try:
            from cupy.cuda import cufft
            self.cufft_version = _eval_or_error(
                lambda: cufft.getVersion(), Exception)
            if full and not is_hip:
                self.cufft_path = _load_and_get_path('cufft')
        except ImportError:
            pass

        # cuRAND
        self.curand_version = _eval_or_error(
            lambda: cupy.cuda.curand.getVersion(),
            Exception)
        if full and not is_hip:
            self.curand_path = _load_and_get_path('curand')

        # cuSOLVER
        self.cusolver_version = _eval_or_error(
            lambda: cupy.cuda.cusolver._getVersion(),
            Exception)
        if full and not is_hip:
            self.cusolver_path = _load_and_get_path('cusolver')

        # cuSPARSE
        self.cusparse_version = '(available)'
        if full:
            self.cusparse_version = _eval_or_error(
                lambda: cupy.cuda.cusparse.getVersion(
                    cupy.cuda.device.get_cusparse_handle()),
                Exception)
            if not is_hip:
                self.cusparse_path = _load_and_get_path('cusparse')

        # NVRTC
        self.nvrtc_version = _eval_or_error(
            lambda: cupy.cuda.nvrtc.getVersion(),
            Exception)
        if full and not is_hip:
            self.nvrtc_path = _load_and_get_path('nvrtc')

        # Thrust
        try:
            import cupy.cuda.thrust as thrust
            self.thrust_version = thrust.get_build_version()
        except ImportError:
            pass

        # CUDA Extra Include Dirs
        if not is_hip:
            try:
                nvrtc_version = cupy.cuda.nvrtc.getVersion()
            except Exception:
                nvrtc_version = None
            if nvrtc_version is None:
                self.cuda_extra_include_dirs = '(NVRTC unavailable)'
            else:
                self.cuda_extra_include_dirs = str(
                    cupy._environment._get_include_dir_from_conda_or_wheel(
                        *nvrtc_version))

        # NCCL
        if cupy._environment._can_attempt_preload('nccl'):
            if full:
                # TODO(leofang): get rid of preloading?
                cupy._environment._preload_library('nccl')
                if not is_hip:
                    self.nccl_path = _load_and_get_path('nccl')
            else:
                self.nccl_build_version = (
                    '(not loaded; try `import cupy.cuda.nccl` first)')
                self.nccl_runtime_version = self.nccl_build_version
        try:
            from cupy_backends.cuda.libs import nccl
            self.nccl_build_version = nccl.get_build_version()
            nccl_runtime_version = nccl.get_version()
            if nccl_runtime_version == 0:
                nccl_runtime_version = '(unknown)'
            self.nccl_runtime_version = nccl_runtime_version
        except ImportError:
            pass

        # CUB
        self.cub_build_version = cupy.cuda.cub.get_build_version()

        try:
            import cupy.cuda.jitify as jitify
            self.jitify_build_version = jitify.get_build_version()
        except ImportError:
            pass

        # cuTENSOR
        try:
            from cupy_backends.cuda.libs import cutensor
            self.cutensor_version = cutensor.get_version()
        except ImportError:
            pass
        if full and not is_hip:
            self.cutensor_path = _load_and_get_path('cutensor')

        # cuSparseLT
        try:
            from cupy_backends.cuda.libs import cusparselt
            self.cusparselt_version = cusparselt.get_build_version()
        except ImportError:
            pass
        if full and not is_hip:
            self.cusparselt_path = _load_and_get_path('cusparseLt')

        # Cython
        self.cython_build_version = cupy._util.cython_build_ver
        try:
            import Cython
            self.cython_version = Cython.__version__
        except ImportError:
            pass

        # NumPy
        self.numpy_version = numpy.version.full_version

        # SciPy
        try:
            import scipy
            self.scipy_version = scipy.version.full_version
        except ImportError:
            pass

    def __str__(self):
        full = self._full

        records = [
            ('OS',  platform.platform()),
            ('Python Version', platform.python_version()),
            ('CuPy Version', self.cupy_version),
            ('CuPy Platform', 'NVIDIA CUDA' if not is_hip else 'AMD ROCm'),
            ('NumPy Version', self.numpy_version),
            ('SciPy Version', self.scipy_version),
            ('Cython Build Version', self.cython_build_version),
            ('Cython Runtime Version', self.cython_version),
            ('CUDA Root', self.cuda_path),
            ('HIPCC PATH' if is_hip else 'NVCC PATH', self.nvcc_path),

            ('CUDA Build Version', self.cuda_build_version),
            ('CUDA Driver Version', self.cuda_driver_version),

            ('CUDA Runtime Version', (
                f'{self.cuda_runtime_version} (linked to CuPy) / '
                f'{self.cuda_local_runtime_version} (locally installed)'
            )),
        ]
        if full and not is_hip:
            records += [(
                'CUDA Runtime Path',
                f'{self.cuda_local_runtime_path} (locally installed)'
            ),]

        records += [
            ('CUDA Extra Include Dirs', self.cuda_extra_include_dirs),
        ]

        ctk_lib_vers = [
            ('cuBLAS Version', self.cublas_version),
            ('cuFFT Version', self.cufft_version),
            ('cuRAND Version', self.curand_version),
            ('cuSOLVER Version', self.cusolver_version),
            ('cuSPARSE Version', self.cusparse_version),
            ('NVRTC Version', self.nvrtc_version),
            ('Thrust Version', self.thrust_version),
            ('CUB Build Version', self.cub_build_version),
            ('Jitify Build Version', self.jitify_build_version),
        ]

        if full and not is_hip:
            ctk_lib_paths = [
                ('cuBLAS Path', self.cublas_path),
                ('cuFFT Path', self.cufft_path),
                ('cuRAND Path', self.curand_path),
                ('cuSOLVER Path', self.cusolver_path),
                ('cuSPARSE Path', self.cusparse_path),
                ('NVRTC Path', self.nvrtc_path),
            ]
            records += _version_and_path(ctk_lib_vers, ctk_lib_paths)
        else:
            records += ctk_lib_vers

        records += [
            ('NCCL Build Version', self.nccl_build_version),
            ('NCCL Runtime Version', self.nccl_runtime_version),
        ]
        if full and not is_hip:
            records += [('NCCL Path', self.nccl_path)]

        records += [
            ('cuTENSOR Version', self.cutensor_version),
        ]
        if full and not is_hip:
            records += [('cuTENSOR Path', self.cutensor_path)]

        records += [
            ('cuSPARSELt Build Version', self.cusparselt_version),
        ]
        if full and not is_hip:
            records += [('cuSPARSELt Path', self.cusparselt_path)]

        device_count = 0
        try:
            device_count = cupy.cuda.runtime.getDeviceCount()
        except cupy.cuda.runtime.CUDARuntimeError as e:
            if 'ErrorNoDevice' not in e.args[0]:
                warnings.warn(f'Failed to detect number of GPUs: {e}')
            # No GPU devices available.
        for device_id in range(device_count):
            with cupy.cuda.Device(device_id) as device:
                props = cupy.cuda.runtime.getDeviceProperties(device_id)
                name = ('Device {} Name'.format(device_id),
                        props['name'].decode())
                pci_bus = ('Device {} PCI Bus ID'.format(device_id),
                           device.pci_bus_id)
                if is_hip:
                    try:
                        arch = props['gcnArchName'].decode()
                    except KeyError:  # ROCm < 3.6.0
                        arch = 'gfx'+str(props['gcnArch'])
                    arch = ('Device {} Arch'.format(device_id), arch)
                else:
                    arch = ('Device {} Compute Capability'.format(device_id),
                            device.compute_capability)
                records += [name, arch, pci_bus]

        width = max([len(r[0]) for r in records]) + 2
        fmt = '{:' + str(width) + '}: {}\n'
        s = io.StringIO()
        for k, v in records:
            s.write(fmt.format(k, v))

        return s.getvalue()


def get_runtime_info(*, full=True):
    return _RuntimeInfo(full=full)


def get_install_info():
    return _InstallInfo()


def _dir_or_none(path):
    """Returns None if path does not exist."""
    if os.path.isdir(path):
        return path
    return None
