import inspect
import io
import os
import platform
import warnings

import numpy

import cupy
import cupy_backends


is_hip = cupy_backends.cuda.api.runtime.is_hip


def _eval_or_error(func, errors):
    # Evaluates `func` and return the result.
    # If an error specified by `errors` occured, it returns a string
    # representing the error.
    try:
        return func()
    except errors as e:
        return repr(e)


class _InstallInfo(object):

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

    # CUDA Driver
    cuda_build_version = None
    cuda_driver_version = None

    # CUDA Runtime
    cuda_runtime_version = None

    # CUDA Toolkit
    cublas_version = None
    cufft_version = None
    curand_version = None
    cusolver_version = None
    cusparse_version = None
    nvrtc_version = None
    thrust_version = None

    # Optional Libraries
    cudnn_build_version = None
    cudnn_version = None
    nccl_build_version = None
    nccl_runtime_version = None
    cub_build_version = None
    jitify_build_version = None
    cutensor_version = None
    cusparselt_version = None
    cython_build_version = None
    cython_version = None

    numpy_version = None
    scipy_version = None

    def __init__(self, *, full=True):
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
            import cuda
            self.cuda_build_version += f' (CUDA Python: {cuda.__version__})'
        self.cuda_driver_version = _eval_or_error(
            cupy.cuda.runtime.driverGetVersion,
            cupy.cuda.runtime.CUDARuntimeError)

        # CUDA Runtime
        self.cuda_runtime_version = _eval_or_error(
            cupy.cuda.runtime.runtimeGetVersion,
            cupy.cuda.runtime.CUDARuntimeError)

        # cuBLAS
        self.cublas_version = '(available)'
        if full:
            self.cublas_version = _eval_or_error(
                lambda: cupy.cuda.cublas.getVersion(
                    cupy.cuda.device.get_cublas_handle()),
                cupy.cuda.cublas.CUBLASError)

        # cuFFT
        self.cufft_version = _eval_or_error(
            cupy.cuda.cufft.getVersion,
            cupy.cuda.cufft.CuFFTError)

        # cuRAND
        self.curand_version = _eval_or_error(
            cupy.cuda.curand.getVersion,
            cupy.cuda.curand.CURANDError)

        # cuSOLVER
        self.cusolver_version = _eval_or_error(
            cupy.cuda.cusolver._getVersion,
            cupy.cuda.cusolver.CUSOLVERError)

        # cuSPARSE
        self.cusparse_version = '(available)'
        if full:
            self.cusparse_version = _eval_or_error(
                lambda: cupy.cuda.cusparse.getVersion(
                    cupy.cuda.device.get_cusparse_handle()),
                cupy.cuda.cusparse.CuSparseError)

        # NVRTC
        self.nvrtc_version = _eval_or_error(
            cupy.cuda.nvrtc.getVersion,
            cupy.cuda.nvrtc.NVRTCError)

        # Thrust
        try:
            import cupy.cuda.thrust as thrust
            self.thrust_version = thrust.get_build_version()
        except ImportError:
            pass

        # cuDNN
        if cupy._environment._can_attempt_preload('cudnn'):
            if full:
                cupy._environment._preload_library('cudnn')
            else:
                self.cudnn_build_version = (
                    '(not loaded; try `import cupy.cuda.cudnn` first)')
                self.cudnn_version = self.cudnn_build_version
        try:
            import cupy_backends.cuda.libs.cudnn as cudnn
            self.cudnn_build_version = cudnn.get_build_version()
            self.cudnn_version = _eval_or_error(
                cudnn.getVersion, cudnn.CuDNNError)
        except ImportError:
            pass

        # NCCL
        try:
            import cupy_backends.cuda.libs.nccl as nccl
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
            import cupy_backends.cuda.libs.cutensor as cutensor
            self.cutensor_version = cutensor.get_version()
        except ImportError:
            pass

        # cuSparseLT
        try:
            import cupy_backends.cuda.libs.cusparselt as cusparselt
            self.cusparselt_version = cusparselt.get_build_version()
        except ImportError:
            pass

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
            ('hipcc PATH' if is_hip else 'nvcc PATH', self.nvcc_path),

            ('CUDA Build Version', self.cuda_build_version),
            ('CUDA Driver Version', self.cuda_driver_version),

            ('CUDA Runtime Version', self.cuda_runtime_version),
        ]

        records += [
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

        records += [
            ('cuDNN Build Version', self.cudnn_build_version),
            ('cuDNN Version', self.cudnn_version),
            ('NCCL Build Version', self.nccl_build_version),
            ('NCCL Runtime Version', self.nccl_runtime_version),
            ('cuTENSOR Version', self.cutensor_version),
            ('cuSPARSELt Build Version', self.cusparselt_version),
        ]

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
