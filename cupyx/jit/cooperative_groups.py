from cupy.cuda import runtime as _runtime
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import BuiltinFunc as _BuiltinFunc
from cupyx.jit._internal_types import Data as _Data


# public interface of this module
__all__ = ['this_grid', 'this_thread_block']


class _ThreadGroup(_cuda_types.TypeBase):
    """ Base class for all cooperative groups. """

    child_type = None

    def __init__(self):
        raise NotImplementedError

    def __str__(self):
        return f'{self.child_type}'

    def _check_cg_include(self, env):
        included = env.generated.cg_include
        if included is False:
            # prepend the header
            env.generated.codes.insert(
                0, "\n#include <cooperative_groups.h>\n"
                   "namespace cg = cooperative_groups;\n")
            env.generated.cg_include = True

    def sync(self, env):
        self._check_cg_include(env)
        return _Data('sync()', _cuda_types.void)


class _GridGroup(_ThreadGroup):

    def __init__(self):
        self.child_type = 'cg::grid_group'

    def is_valid(self, env):
        self._check_cg_include(env)
        return _Data('is_valid()', _cuda_types.bool_)

    def sync(self, env):
        # when this methond is called, we need to use the cooperative
        # launch API
        env.generated.enable_cg = True
        return super().sync(env)

    def thread_rank(self, env):
        return _Data('thread_rank()', _cuda_types.uint64)

    def block_rank(self, env):
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("block_rank() is supported on CUDA 11.6+")
        return _Data('block_rank()', _cuda_types.uint64)

    def num_threads(self, env):
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("num_threads() is supported on CUDA 11.6+")
        return _Data('num_threads()', _cuda_types.uint64)

    def num_blocks(self, env):
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("num_blocks() is supported on CUDA 11.6+")
        return _Data('num_blocks()', _cuda_types.uint64)

    def dim_blocks(self, env):
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("dim_blocks() is supported on CUDA 11.6+")
        from cupyx.jit._interface import _Dim3  # avoid circular import
        return _Data('dim_blocks()', _Dim3())

    def block_index(self, env):
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("block_index() is supported on CUDA 11.6+")
        from cupyx.jit._interface import _Dim3  # avoid circular import
        return _Data('block_index()', _Dim3())

    def size(self, env):
        # despite it is an alias of num_threads, we need it for earlier 11.x
        return _Data('size()', _cuda_types.uint64)

    def group_dim(self, env):
        # despite it is an alias of dim_blocks, we need it for earlier 11.x
        from cupyx.jit._interface import _Dim3  # avoid circular import
        return _Data('group_dim()', _Dim3())


class _ThreadBlockGroup(_ThreadGroup):

    def __init__(self):
        self.child_type = 'cg::thread_block'

    def thread_rank(self, env):
        return _Data('thread_rank()', _cuda_types.uint32)

    def group_index(self, env):
        from cupyx.jit._interface import _Dim3  # avoid circular import
        return _Data('group_index()', _Dim3())

    def thread_index(self, env):
        from cupyx.jit._interface import _Dim3  # avoid circular import
        return _Data('thread_index()', _Dim3())

    def dim_threads(self, env):
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("dim_threads() is supported on CUDA 11.6+")
        from cupyx.jit._interface import _Dim3  # avoid circular import
        return _Data('dim_threads()', _Dim3())

    def num_threads(self, env):
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("num_threads() is supported on CUDA 11.6+")
        return _Data('num_threads()', _cuda_types.uint32)

    def size(self, env):
        # despite it is an alias of num_threads, we need it for earlier 11.x
        return _Data('size()', _cuda_types.uint32)

    def group_dim(self, env):
        # despite it is an alias of dim_threads, we need it for earlier 11.x
        from cupyx.jit._interface import _Dim3  # avoid circular import
        return _Data('group_dim()', _Dim3())


class _ThisCgGroup(_BuiltinFunc):

    def __init__(self, group_type):
        self.group_type = group_type

    def call_const(self, env):
        if _runtime.is_hip:
            raise RuntimeError('cooperative group is not supported on HIP')
        if self.group_type == 'grid':
            cg_type = _GridGroup()
        elif self.group_type == 'thread_block':
            cg_type = _ThreadBlockGroup()
        else:
            raise NotImplementedError
        return _Data(f'cg::this_{self.group_type}()', cg_type)


this_grid = _ThisCgGroup('grid')
this_thread_block = _ThisCgGroup('thread_block')
