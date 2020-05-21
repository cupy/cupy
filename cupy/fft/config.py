from cupy import util


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
    util.experimental('cupy.fft.config.set_cufft_gpus')
    global _devices

    if isinstance(gpus, int):
        devs = [i for i in range(gpus)]
    elif isinstance(gpus, list):
        devs = gpus
    else:
        raise ValueError("gpus must be an int or a list of int.")
    if len(devs) <= 1:
        raise ValueError("Must use at least 2 GPUs.")

    _devices = devs
