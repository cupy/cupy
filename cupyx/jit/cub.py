from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit import _internal_types
from cupy_backends.cuda.api import runtime as _runtime


class _ClassTemplate:

    def __init__(self, class_type):
        self._class_type = class_type
        self.__doc__ = self._class_type.__doc__

    def __getitem__(self, args):
        if isinstance(args, tuple):
            return self._class_type(*args)
        else:
            return self._class_type(args)


def _include_cub(env):
    if _runtime.is_hip:
        env.generated.add_code('#include <hipcub/hipcub.hpp>')
        env.generated.backend = 'nvcc'
    else:
        # Including the master header (cub/cub.cuh) would break NVRTC, because
        # it implicitly includes many Thrust headers that would give Jitify a
        # hard time. We only include what we need (for block-/warp- level
        # algorithms).
        # WAR: warp_reduce.cuh is implicitly included.
        env.generated.add_code('#include <cub/block/block_reduce.cuh>')
        env.generated.backend = 'nvrtc'
        env.generated.jitify = True


def _get_cub_namespace():
    return 'hipcub' if _runtime.is_hip else 'cub'


class _TempStorageType(_cuda_types.TypeBase):

    def __init__(self, parent_type):
        assert isinstance(parent_type, _CubReduceBaseType)
        self.parent_type = parent_type
        super().__init__()

    def __str__(self) -> str:
        return f'typename {self.parent_type}::TempStorage'


class _CubReduceBaseType(_cuda_types.TypeBase):

    def _instantiate(self, env, temp_storage) -> _internal_types.Data:
        _include_cub(env)
        if temp_storage.ctype != self.TempStorage:
            raise TypeError(
                f'Invalid temp_storage type {temp_storage.ctype}. '
                f'({self.TempStorage} is expected.)')
        return _internal_types.Data(f'{self}({temp_storage.code})', self)

    @_internal_types.wraps_class_method
    def Sum(self, env, instance, input) -> _internal_types.Data:
        if input.ctype != self.T:
            raise TypeError(
                f'Invalid input type {input.ctype}. ({self.T} is expected.)')
        return _internal_types.Data(
            f'{instance.code}.Sum({input.code})', input.ctype)

    @_internal_types.wraps_class_method
    def Reduce(self, env, instance, input, reduction_op):
        if input.ctype != self.T:
            raise TypeError(
                f'Invalid input type {input.ctype}. ({self.T} is expected.)')
        return _internal_types.Data(
            f'{instance.code}.Reduce({input.code}, {reduction_op.code})',
            input.ctype)


class _WarpReduceType(_CubReduceBaseType):

    def __init__(self, T) -> None:
        self.T = _cuda_typerules.to_ctype(T)
        self.TempStorage = _TempStorageType(self)
        super().__init__()

    def __str__(self) -> str:
        namespace = _get_cub_namespace()
        return f'{namespace}::WarpReduce<{self.T}>'


class _BlockReduceType(_CubReduceBaseType):

    def __init__(self, T, BLOCK_DIM_X: int) -> None:
        self.T = _cuda_typerules.to_ctype(T)
        self.BLOCK_DIM_X = BLOCK_DIM_X
        self.TempStorage = _TempStorageType(self)
        super().__init__()

    def __str__(self) -> str:
        namespace = _get_cub_namespace()
        return f'{namespace}::BlockReduce<{self.T}, {self.BLOCK_DIM_X}>'


WarpReduce = _ClassTemplate(_WarpReduceType)
BlockReduce = _ClassTemplate(_BlockReduceType)


class _CubFunctor(_internal_types.BuiltinFunc):

    def __init__(self, name):
        namespace = _get_cub_namespace()
        self.fname = f'{namespace}::{name}()'

    def call_const(self, env):
        return _internal_types.Data(
            self.fname, _cuda_types.Unknown(label='cub_functor'))


Sum = _CubFunctor('Sum')
Max = _CubFunctor('Max')
Min = _CubFunctor('Min')
