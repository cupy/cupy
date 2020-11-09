from cupyx.jit import _compile


class _CudaFunction:
    """JIT cupy function object
    """

    def __init__(self, func, mode, device=False, inline=False):
        self.attributes = []

        if device:
            self.attributes.append('__device__')
        else:
            self.attributes.append('__global__')

        if inline:
            self.attributes.append('inline')

        self.name = getattr(func, 'name', func.__name__)
        self.func = func
        self.mode = mode

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def _emit_code_from_types(self, in_types, ret_type=None):
        return _compile.transpile(
            self.func, self.attributes, self.mode, in_types, ret_type)
