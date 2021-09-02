class Backend:
    """Interface for different communication backends.
    """
    def all_reduce(self, in_array, out_array, op='sum', stream=None):
        raise NotImplementedError(
            'Current Backend does not implement')

    def reduce(self, in_array, out_array, root=0, op='sum', stream=None):
        raise NotImplementedError(
            'Current Backend does not implement')

    def broadcast(self, in_array, root=0, stream=None):
        raise NotImplementedError(
            'Current Backend does not implement')

    def reduce_scatter(
            self, in_array, out_array, count, op='sum', stream=None):
        raise NotImplementedError(
            'Current Backend does not implement')

    def all_gather(self, in_array, out_array, count, stream=None):
        raise NotImplementedError(
            'Current Backend does not implement')

    def send(self, array, peer, stream=None):
        raise NotImplementedError(
            'Current Backend does not implement')

    def recv(self, out_array, peer, stream=None):
        raise NotImplementedError(
            'Current Backend does not implement')

    def send_recv(self, in_array, out_array, peer, stream=None):
        raise NotImplementedError(
            'Current Backend does not implement')

    def scatter(self, in_array, out_array, root=0, stream=None):
        raise NotImplementedError(
            'Current Backend does not implement')

    def gather(self, in_array, out_array, root=0, stream=None):
        raise NotImplementedError(
            'Current Backend does not implement')

    def all_to_all(self, in_array, out_array, stream=None):
        raise NotImplementedError(
            'Current Backend does not implement')

    def barrier(self):
        raise NotImplementedError(
            'Current Backend does not implement')
