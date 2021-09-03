import os

import cupy.cuda.nccl

from cupyx.distributed import _store
from cupyx.distributed._nccl_comm import NCCLBackend


_backends = {'nccl': NCCLBackend}


def init_process_group(
        n_devices, rank, *, backend='nccl', host=None, port=None):
    """Start `cupyx.distributed` and obtain a communicator.

    This call initializes the distributed environment, it needs to be
    called for every process that is involved in the communications.

    We only allow to run a single GPU per returned communication
    object, being the user the responsible of setting the appropiated gpu
    to be used.

    Currently the user needs to specify each process rank and the total
    number of processes, and start all the processes in different hosts
    manually.

    The process with rank 0 will spawn a TCP server using a
    subprocess that listens in the port indicated by
    the env var `CUPYX_DISTRIBUTED_PORT`, the rank 0 must be executed
    in the host determined by the env var `CUPYX_DISTRIBUTED_HOST`.
    In case their values are not specified, `'127.0.0.1'` and `12345` will be
    used by default.

    Add example showing how to use it with two processes here:
    >>>

    Args:
        n_devices (int): Total number of devices that will be used in the
            distributed execution.
        rank (int): Unique id of the GPU that the communicator is associated to
            its value needs to be `0 <= rank < n_devices`.
        backend (str): Backend to use for the communications. Optional,
            defaults to `"nccl"`.
        host (str): host address for the process rendezvous on initialization
            defaults to `None`.
        port (int): port for the process rendezvous on initialization
            defaults to `None`.
    Returns:
        Backend: object used to perform communications, adheres to the
            :class:`~cupyx.distributed.Backend` specification:
    """
    if n_devices <= 0:
        raise ValueError(f'Invalid number of devices {n_devices}')
    if not (0 <= rank < n_devices):
        raise ValueError(f'Invalid number of rank {rank} {n_devices}')
    if backend not in _backends:
        raise ValueError(f'{backend} is not supported')
    if not cupy.cuda.nccl.available:
        raise RuntimeError('NCCL is not available')
    if host is None:
        host = os.environ.get('CUPYX_DISTRIBUTED_HOST', _store._DEFAULT_HOST)
    if port is None:
        port = int(os.environ.get(
            'CUPYX_DISTRIBUTED_PORT', _store._DEFAULT_PORT))

    return _backends[backend](n_devices, rank, host, port)
