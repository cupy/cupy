import inspect
import io
import os
import platform

import numpy

import cupy
import cupy_backends

try:
    import cupy.cuda.thrust as thrust
except ImportError:
    thrust = None

try:
    import cupy_backends.cuda.libs.cudnn as cudnn
except ImportError:
    cudnn = None

try:
    import cupy.cuda.nccl as nccl
except ImportError:
    nccl = None

try:
    import cupy.cuda.cub as cub
except ImportError:
    cub = None

try:
    import cupy_backends.cuda.libs.cutensor as cutensor
except ImportError:
    cutensor = None

try:
    import scipy
except ImportError:
    scipy = None

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


class _RuntimeInfo(object):

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
    cutensor_version = None
    cython_build_version = None

    numpy_version = None
    scipy_version = None

    def __init__(self):
        self.cupy_version = cupy.__version__

        if not is_hip:
            self.cuda_path = cupy.cuda.get_cuda_path()
        else:
            self.cuda_path = cupy._environment.get_rocm_path()

        self.cuda_build_version = cupy.cuda.driver.get_build_version()
        self.cuda_driver_version = _eval_or_error(
            cupy.cuda.runtime.driverGetVersion,
            cupy.cuda.runtime.CUDARuntimeError)

        self.cuda_runtime_version = _eval_or_error(
            cupy.cuda.runtime.runtimeGetVersion,
            cupy.cuda.runtime.CUDARuntimeError)

        self.cublas_version = _eval_or_error(
            lambda: cupy.cuda.cublas.getVersion(
                cupy.cuda.device.get_cublas_handle()),
            cupy.cuda.cublas.CUBLASError)
        self.cufft_version = _eval_or_error(
            cupy.cuda.cufft.getVersion,
            cupy.cuda.cufft.CuFFTError)
        self.curand_version = _eval_or_error(
            cupy.cuda.curand.getVersion,
            cupy.cuda.curand.CURANDError)
        self.cusolver_version = _eval_or_error(
            cupy.cuda.cusolver._getVersion,
            cupy.cuda.cusolver.CUSOLVERError)
        self.cusparse_version = _eval_or_error(
            lambda: cupy.cuda.cusparse.getVersion(
                cupy.cuda.device.get_cusparse_handle()),
            cupy.cuda.cusparse.CuSparseError)
        self.nvrtc_version = _eval_or_error(
            cupy.cuda.nvrtc.getVersion,
            cupy.cuda.nvrtc.NVRTCError)

        if thrust is not None:
            self.thrust_version = thrust.get_build_version()

        if cudnn is not None:
            self.cudnn_build_version = cudnn.get_build_version()
            self.cudnn_version = _eval_or_error(
                cudnn.getVersion, cudnn.CuDNNError)

        if nccl is not None:
            self.nccl_build_version = nccl.get_build_version()
            nccl_runtime_version = nccl.get_version()
            if nccl_runtime_version == 0:
                nccl_runtime_version = '(unknown)'
            self.nccl_runtime_version = nccl_runtime_version

        if cub is not None:
            self.cub_build_version = cub.get_build_version()

        if cutensor is not None:
            self.cutensor_version = cutensor.get_version()

        self.cython_build_version = cupy._util.cython_build_ver

        self.numpy_version = numpy.version.full_version
        if scipy is not None:
            self.scipy_version = scipy.version.full_version

    def __str__(self):
        records = [
            ('OS',  platform.platform()),
            ('CuPy Version', self.cupy_version),
            ('NumPy Version', self.numpy_version),
            ('SciPy Version', self.scipy_version),
            ('Cython Build Version', self.cython_build_version),
            ('CUDA Root', self.cuda_path),

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
        ]

        records += [
            ('cuDNN Build Version', self.cudnn_build_version),
            ('cuDNN Version', self.cudnn_version),
            ('NCCL Build Version', self.nccl_build_version),
            ('NCCL Runtime Version', self.nccl_runtime_version),
            ('cuTENSOR Version', self.cutensor_version),
        ]

        for device_id in range(cupy.cuda.runtime.getDeviceCount()):
            with cupy.cuda.Device(device_id) as device:
                props = cupy.cuda.runtime.getDeviceProperties(device_id)
                records += [
                    ('Device {} Name'.format(device_id),
                     props['name'].decode('utf-8')),
                    ('Device {} Compute Capability'.format(device_id),
                     device.compute_capability),
                ]

        width = max([len(r[0]) for r in records]) + 2
        fmt = '{:' + str(width) + '}: {}\n'
        s = io.StringIO()
        for k, v in records:
            s.write(fmt.format(k, v))

        return s.getvalue()


def get_runtime_info():
    return _RuntimeInfo()


def get_install_info():
    return _InstallInfo()


def _dir_or_none(path):
    """Returns None if path does not exist."""
    if os.path.isdir(path):
        return path
    return None
