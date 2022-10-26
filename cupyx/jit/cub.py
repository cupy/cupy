from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit import _internal_types


class _ClassTemplate:

    def __init__(self, class_type):
        self._class_type = class_type
        self.__doc__ = self._class_type.__doc__

    def __getitem__(self, template_type):
        return self._class_type(_cuda_typerules.to_ctype(template_type))


def _include_cub(env):
    env.generated.add_code('#include <cub/cub.cuh>')
    env.generated.backend = 'nvcc'


class _TempStorageType(_cuda_types.TypeBase):

    def __init__(self, parent_type):
        assert isinstance(parent_type, _WarpReduceType)
        self.parent_type = parent_type
        super().__init__()

    def __str__(self) -> str:
        return f'typename {self.parent_type}::TempStorage'


class _WarpReduceType(_cuda_types.TypeBase):

    def __init__(self, child_type) -> None:
        self.child_type = child_type
        self.TempStorage = _TempStorageType(self)
        super().__init__()

    def __str__(self) -> str:
        return f'cub::WarpReduce<{self.child_type}>'

    def _instantiate(self, env, temp_storage) -> _internal_types.Data:
        _include_cub(env)
        return _internal_types.Data(f'{self}({temp_storage.code})', self)

    @_internal_types.wraps_class_method
    def Sum(self, env, instance, input) -> _internal_types.Data:
        return _internal_types.Data(
            f'{instance.code}.Sum({input.code})', input.ctype)


WarpReduce = _ClassTemplate(_WarpReduceType)
