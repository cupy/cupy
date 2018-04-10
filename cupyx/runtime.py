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


class _RuntimeInfo(object):

    cupy_version = None
    cuda_path = None
    cuda_build_version = None
    cuda_driver_version = None
    cuda_runtime_version = None
    cudnn_build_version = None
    cudnn_version = None
    nccl_build_version = None

    def __init__(self):
        self.cupy_version = cupy.__version__

        self.cuda_path = cupy.cuda.get_cuda_path()
        self.cuda_build_version = cupy.cuda.driver.get_build_version()
        self.cuda_driver_version = cupy.cuda.runtime.driverGetVersion()
        self.cuda_runtime_version = cupy.cuda.runtime.runtimeGetVersion()

        if cudnn is not None:
            self.cudnn_build_version = cudnn.get_build_version()
            self.cudnn_version = cudnn.getVersion()

        if nccl is not None:
            self.nccl_build_version = nccl.get_version()

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
        ]
        width = max([len(r[0]) for r in records]) + 2
        fmt = '{:' + str(width) + '}: {}\n'
        s = six.StringIO()
        for k, v in records:
            s.write(fmt.format(k, v))

        return s.getvalue()


def get_runtime_info():
    return _RuntimeInfo()
