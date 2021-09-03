from cupyx.distributed import _store


class Backend:
    """Interface for different communication backends.

    The `Backend` interface provides methods to perform collective and
    point-to-point communications.

    A backend object is associated with a single GPU. In the case of managing
    multiple GPUs in a single process, the backend associated GPU must be
    explicitly set before calling any of the backend operations.

    Args:
        n_devices (int): Total number of devices that will be used in the
            distributed execution.
        rank (int): Unique id of the GPU that the communicator is associated to
            its value needs to be `0 <= rank < n_devices`.
        host (str, optional): host address for the process rendezvous on
            initialization. Defaults to `"127.0.0.1"`.
        port (int, optional): port used for the process rendezvous on
            initialization. Defaults to `12345`.
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
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array where the result with be stored.
            op (str): reduction operation, can be one of
                ('sum', 'prod', 'min' 'max'), arrays of complex type only
                support `'sum'`. Defaults to `'sum'`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        raise NotImplementedError(
            'Current Backend does not implement all_reduce')

    def reduce(self, in_array, out_array, root=0, op='sum', stream=None):
        """Performs a reduce operation.

        Args:
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array where the result with be stored.
                will only be modified by the `root` process.
            root (int, optional): rank of the process that will perform the
                reduction. Defaults to `0`.
            op (str): reduction operation, can be one of
                ('sum', 'prod', 'min' 'max'), arrays of complex type only
                support `'sum'`. Defaults to `'sum'`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        raise NotImplementedError(
            'Current Backend does not implement reduce')

    def broadcast(self, in_array, root=0, stream=None):
        """Performs a broadcast operation.

        Args:
            in_array (cupy.ndarray): array to be sent for `root` rank.
                Other ranks will receive the broadcast data here.
            root (int, optional): rank of the process that will send the
                broadcast. Defaults to `0`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        raise NotImplementedError(
            'Current Backend does not implement broadcast')

    def reduce_scatter(
            self, in_array, out_array, count, op='sum', stream=None):
        """Performs a reduce scatter operation.

        Args:
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array where the result with be stored.
            count (int): Number of elements to send to each rank.
            op (str): reduction operation, can be one of
                ('sum', 'prod', 'min' 'max'), arrays of complex type only
                support `'sum'`. Defaults to `'sum'`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        raise NotImplementedError(
            'Current Backend does not implement reduce_scatter')

    def all_gather(self, in_array, out_array, count, stream=None):
        """Performs an all gather operation.

        Args:
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array where the result with be stored.
            count (int): Number of elements to send to each rank.
            op (str): reduction operation, can be one of
                ('sum', 'prod', 'min' 'max'), arrays of complex type only
                support `'sum'`. Defaults to `'sum'`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        raise NotImplementedError(
            'Current Backend does not implement all_gather')

    def send(self, array, peer, stream=None):
        """Performs a send operation.

        Args:
            array (cupy.ndarray): array to be sent.
            peer (int): rank of the process `array` will be sent to.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        raise NotImplementedError(
            'Current Backend does not implement send')

    def recv(self, out_array, peer, stream=None):
        """Performs a receive operation.

        Args:
            array (cupy.ndarray): array used to receive data.
            peer (int): rank of the process `array` will be received from.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        raise NotImplementedError(
            'Current Backend does not implement recv')

    def send_recv(self, in_array, out_array, peer, stream=None):
        """Performs a send and receive operation.

        Args:
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array used to receive data.
            peer (int): rank of the process to send `in_array` and receive
                `out_array`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        raise NotImplementedError(
            'Current Backend does not implement send_recv')

    def scatter(self, in_array, out_array, root=0, stream=None):
        """Performs a scatter operation.

        Args:
            in_array (cupy.ndarray): array to be sent. Its shape must be
                `(total_ranks, ...)`.
            out_array (cupy.ndarray): array where the result with be stored.
            root (int): rank that will send the `in_array` to other ranks.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        raise NotImplementedError(
            'Current Backend does not implement scatter')

    def gather(self, in_array, out_array, root=0, stream=None):
        """Performs a gather operation.

        Args:
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array where the result with be stored.
                Its shape must be `(total_ranks, ...)`.
            root (int): rank that will receive the `in_array`s
                from other ranks.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        raise NotImplementedError(
            'Current Backend does not implement gather')

    def all_to_all(self, in_array, out_array, stream=None):
        """Performs an all to all operation.

        Args:
            in_array (cupy.ndarray): array to be sent. Its shape must be
                `(total_ranks, ...)`.
            out_array (cupy.ndarray): array where the result with be stored.
                Its shape must be `(total_ranks, ...)`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        raise NotImplementedError(
            'Current Backend does not implement all_to_all')

    def barrier(self):
        """Performs a barrier operation.

        The barrier is done in the cpu and is a explicit synchronization
        mechanism that halts the thread progression.
        """
        raise NotImplementedError(
            'Current Backend does not implement barrier')
