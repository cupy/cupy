import os

from cupy.cuda import nccl

from cupyx.distributed import _store
from cupyx.distributed._nccl_comm import NCCLBackend


_backends = {'nccl': NCCLBackend}


def init_process_group(
        n_devices, rank, *, backend='nccl', host=None, port=None,
        use_mpi=False):
    """Start `cupyx.distributed` and obtain a communicator.

    This call initializes the distributed environment, it needs to be
    called for every process that is involved in the communications.

    A single device per returned communication is only allowed. It is the user
    responsibility of setting the appropiated gpu to be used before creating
    and using the communicator.

    Currently the user needs to specify each process rank and the total
    number of processes, and start all the processes in different hosts
    manually.

    The process with rank 0 will spawn a TCP server using a
    subprocess that listens in the port indicated by
    the env var `CUPYX_DISTRIBUTED_PORT`, the rank 0 must be executed
    in the host determined by the env var `CUPYX_DISTRIBUTED_HOST`.
    In case their values are not specified, `'127.0.0.1'` and `13333` will be
    used by default.

    Note that this feature is expected to be used within a trusted cluster
    environment.

    Example:

        >>> import cupy
        >>> def process_0():
        ...     import cupyx.distributed
        ...     cupy.cuda.Device(0).use()
        ...     comm = cupyx.distributed.init_process_group(2, 0)
        ...     array = cupy.ones(1)
        ...     comm.broadcast(array, 0)
        ...
        >>> def process_1():
        ...     import cupyx.distributed
        ...     cupy.cuda.Device(1).use()
        ...     comm = cupyx.distributed.init_process_group(2, 1)
        ...     array = cupy.zeros(1)
        ...     comm.broadcast(array, 0)
        ...     cupy.equal(array, cupy.ones(1))

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
        use_mpi (bool): if ``False``, it avoids using MPI for synchronization
            and uses the provided TCP server for exchanging CPU only
            information.
            defaults to `False`.
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
    if backend == 'nccl' and not nccl.available:
        raise RuntimeError('NCCL is not available')
    if host is None:
        host = os.environ.get('CUPYX_DISTRIBUTED_HOST', _store._DEFAULT_HOST)
    if port is None:
        port = int(os.environ.get(
            'CUPYX_DISTRIBUTED_PORT', _store._DEFAULT_PORT))

    return _backends[backend](n_devices, rank, host, port, use_mpi)
