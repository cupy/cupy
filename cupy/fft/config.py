global enable_nd_planning
enable_nd_planning = True

global use_multi_gpus
use_multi_gpus=False

global _devices
_devices=None

def set_cufft_gpus(gpus):
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
