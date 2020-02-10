import os
import shutil


_cuda_path = ()
_nvcc_path = ()


def get_cuda_path():
    # Returns the CUDA installation path or None if not found.
    global _cuda_path
    if _cuda_path == ():
        _cuda_path = _get_cuda_path()
    return _cuda_path


def get_nvcc_path():
    # Returns the path to the nvcc command or None if not found.
    global _nvcc_path
    if _nvcc_path == ():
        _nvcc_path = _get_nvcc_path()
    return _nvcc_path


def _get_cuda_path():
    # Use environment variable
    cuda_path = os.environ.get('CUDA_PATH', '')  # Nvidia default on Windows
    if os.path.exists(cuda_path):
        return cuda_path

    # Use nvcc path
    nvcc_path = shutil.which('nvcc')
    if nvcc_path is not None:
        return os.path.dirname(os.path.dirname(nvcc_path))

    # Use typical path
    if os.path.exists('/usr/local/cuda'):
        return '/usr/local/cuda'

    return None


def _get_nvcc_path():
    # Directly lookup PATH
    nvcc_path = shutil.which('nvcc')
    if nvcc_path is not None:
        return nvcc_path

    # Lookup <CUDA>/bin
    cuda_path = get_cuda_path()
    if cuda_path is None:
        return None

    return shutil.which(
        'nvcc',
        path=os.path.join(cuda_path, 'bin'))
