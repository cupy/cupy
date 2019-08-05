import inspect
import os
import six

import cupy

try:
    import cupy.cuda.cudnn as cudnn
except ImportError:
    cudnn = None

try:
    import cupy.cuda.nccl as nccl
except ImportError:
    nccl = None


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
    cuda_build_version = None
    cuda_driver_version = None
    cuda_runtime_version = None
    cudnn_build_version = None
    cudnn_version = None
    nccl_build_version = None
    nccl_runtime_version = None

    def __init__(self):
        self.cupy_version = cupy.__version__

        self.cuda_path = cupy.cuda.get_cuda_path()
        self.cuda_build_version = cupy.cuda.driver.get_build_version()
        self.cuda_driver_version = _eval_or_error(
            cupy.cuda.runtime.driverGetVersion,
            cupy.cuda.runtime.CUDARuntimeError)
        self.cuda_runtime_version = _eval_or_error(
            cupy.cuda.runtime.runtimeGetVersion,
            cupy.cuda.runtime.CUDARuntimeError)

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

    def __str__(self):
        records = [
            ('CuPy Version', self.cupy_version),
            ('CUDA Root', self.cuda_path),
            ('CUDA Build Version', self.cuda_build_version),
            ('CUDA Driver Version', self.cuda_driver_version),
            ('CUDA Runtime Version', self.cuda_runtime_version),
            ('cuDNN Build Version', self.cudnn_build_version),
            ('cuDNN Version', self.cudnn_version),
            ('NCCL Build Version', self.nccl_build_version),
            ('NCCL Runtime Version', self.nccl_runtime_version),
        ]
        width = max([len(r[0]) for r in records]) + 2
        fmt = '{:' + str(width) + '}: {}\n'
        s = six.StringIO()
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
