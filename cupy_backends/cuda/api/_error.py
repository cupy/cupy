class _CudaErrorBase(RuntimeError):

    def __init__(self, *args, status=None, **kwargs):
        self.status = status
        self._infos = []
        if status is not None:
            self._init_from_status_code(status)
        else:
            super(_CudaErrorBase, self).__init__(*args, **kwargs)

    def _init_from_status_code(self, status):
        raise NotImplementedError

    def _init_from_msg(self, name, msg):
        super(_CudaErrorBase, self).__init__('{}: {}'.format(name, msg))

    def add_info(self, info):
        assert isinstance(info, str)
        self._infos.append(info)

    def add_infos(self, infos):
        assert isinstance(infos, list)
        self._infos.extend(infos)

    def __str__(self):
        base = super(_CudaErrorBase, self).__str__()
        return base + ''.join(
            '\n  ' + info for info in self._infos)

    def __reduce__(self):
        if self.status is not None:
            return (type(self), (self.status,))
        else:
            return super(_CudaErrorBase, self).__reduce__()
