from cupyx.distributed import _store


class Backend:
    """Interface for different communication backends.


    Args:
        n_devices (int): Total number of devices that will be used in the
            distributed execution.
        rank (int): Unique id of the GPU that the communicator is associated to
            its value needs to be `0 <= rank < n_devices`.
        host (str): host address for the process rendezvous on initialization
            defaults to `"127.0.0.1"`.
        port (int): port for the process rendezvous on initialization
            defaults to `12345`.
    """
    def __init__(self, n_devices, rank,
                 host=_store._DEFAULT_HOST, port=_store._DEFAULT_PORT):
        self._n_devices = n_devices
        self.rank = rank
        self._store_proxy = _store.TCPStoreProxy(host, port)
        if rank == 0:
            self._store = _store.TCPStore(n_devices)

    def all_reduce(self, in_array, out_array, op='sum', stream=None):
        """Performs an all reduce operation.

        Args:
            in_array (cupy.ndarray):
            out_array (cupy.ndarray):
            op (str):
            stream (cupy.cuda.Stream):
        """
        raise NotImplementedError(
            'Current Backend does not implement')

    def reduce(self, in_array, out_array, root=0, op='sum', stream=None):
        """Performs a reduce operation.

        Args:
            in_array (cupy.ndarray):
            out_array (cupy.ndarray):
            op (str):
            stream (cupy.cuda.Stream):
        """
        raise NotImplementedError(
            'Current Backend does not implement')

    def broadcast(self, in_array, root=0, stream=None):
        """Performs a broadcast operation.

        Args:
            in_array (cupy.ndarray):
            out_array (cupy.ndarray):
            op (str):
            stream (cupy.cuda.Stream):
        """
        raise NotImplementedError(
            'Current Backend does not implement')

    def reduce_scatter(
            self, in_array, out_array, count, op='sum', stream=None):
        """Performs a reduce scatter operation.

        Args:
            in_array (cupy.ndarray):
            out_array (cupy.ndarray):
            op (str):
            stream (cupy.cuda.Stream):
        """
        raise NotImplementedError(
            'Current Backend does not implement')

    def all_gather(self, in_array, out_array, count, stream=None):
        """Performs an all gather operation.

        Args:
            in_array (cupy.ndarray):
            out_array (cupy.ndarray):
            op (str):
            stream (cupy.cuda.Stream):
        """
        raise NotImplementedError(
            'Current Backend does not implement')

    def send(self, array, peer, stream=None):
        """Performs a send operation.

        Args:
            in_array (cupy.ndarray):
            out_array (cupy.ndarray):
            op (str):
            stream (cupy.cuda.Stream):
        """
        raise NotImplementedError(
            'Current Backend does not implement')

    def recv(self, out_array, peer, stream=None):
        """Performs a receive operation.

        Args:
            in_array (cupy.ndarray):
            out_array (cupy.ndarray):
            op (str):
            stream (cupy.cuda.Stream):
        """
        raise NotImplementedError(
            'Current Backend does not implement')

    def send_recv(self, in_array, out_array, peer, stream=None):
        """Performs a send and receive operation.

        Args:
            in_array (cupy.ndarray):
            out_array (cupy.ndarray):
            op (str):
            stream (cupy.cuda.Stream):
        """
        raise NotImplementedError(
            'Current Backend does not implement')

    def scatter(self, in_array, out_array, root=0, stream=None):
        """Performs an all reduce operation.

        Args:
            in_array (cupy.ndarray):
            out_array (cupy.ndarray):
            op (str):
            stream (cupy.cuda.Stream):
        """
        raise NotImplementedError(
            'Current Backend does not implement')

    def gather(self, in_array, out_array, root=0, stream=None):
        """Performs a gather operation.

        Args:
            in_array (cupy.ndarray):
            out_array (cupy.ndarray):
            op (str):
            stream (cupy.cuda.Stream):
        """
        raise NotImplementedError(
            'Current Backend does not implement')

    def all_to_all(self, in_array, out_array, stream=None):
        """Performs an all to all operation.

        Args:
            in_array (cupy.ndarray):
            out_array (cupy.ndarray):
            op (str):
            stream (cupy.cuda.Stream):
        """
        raise NotImplementedError(
            'Current Backend does not implement')

    def barrier(self):
        """Performs a barrier operation
        """
        raise NotImplementedError(
            'Current Backend does not implement')
