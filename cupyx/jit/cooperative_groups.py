from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import BuiltinFunc
from cupyx.jit._internal_types import Data


# public interface of this module
__all__ = ['this_grid', 'this_thread_block']


class ThreadGroup(_cuda_types.TypeBase):
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
            env.generated.codes.insert(0,
                "\n#include <cooperative_groups.h>\n"
                "namespace cg = cooperative_groups;\n")
            env.generated.cg_include = True

    def sync(self, env):
        self._check_cg_include(env)
        return Data('sync()', _cuda_types.void)


class GridGroup(ThreadGroup):

    def __init__(self):
        self.child_type = 'cg::grid_group'

    def is_valid(self, env):
        self._check_cg_include(env)
        return Data('is_valid()', _cuda_types.bool_)

    def sync(self, env):
        # when this methond is called, we need to use the cooperative
        # launch API
        env.generated.enable_cg = True
        return super().sync(env)

    def thread_rank(self, env):
        return Data('thread_rank()', _cuda_types.uint64)

    def block_rank(self, env):
        return Data('block_rank()', _cuda_types.uint64)

    def num_threads(self, env):
        return Data('num_threads()', _cuda_types.uint64)

    def num_blocks(self, env):
        return Data('num_blocks()', _cuda_types.uint64)

    def dim_blocks(self, env):
        from cupyx.jit._interface import _Dim3  # avoid circular import
        return Data('dim_blocks()', _Dim3())

    def block_index(self, env):
        from cupyx.jit._interface import _Dim3  # avoid circular import
        return Data('block_index()', _Dim3())

    def size(self, env):
        return self.num_threads(env)

    def grid_dim(self, env):
        return self.dim_blocks(env)


class ThreadBlockGroup(ThreadGroup):

    def __init__(self):
        self.child_type = 'cg::thread_block'

    def thread_rank(self, env):
        return Data('thread_rank()', _cuda_types.uint32)

    def group_index(self, env):
        from cupyx.jit._interface import _Dim3  # avoid circular import
        return Data('group_index()', _Dim3())

    def thread_index(self, env):
        from cupyx.jit._interface import _Dim3  # avoid circular import
        return Data('thread_index()', _Dim3())

    def dim_threads(self, env):
        from cupyx.jit._interface import _Dim3  # avoid circular import
        return Data('dim_threads()', _Dim3())

    def num_threads(self, env):
        return Data('num_threads()', _cuda_types.uint32)

    def size(self, env):
        return self.num_threads(env)

    def group_dim(self, env):
        return self.dim_threads(env)


class ThisCgGroup(BuiltinFunc):

    def __init__(self, group_type):
        self.group_type = group_type

    def call_const(self, env):
        if self.group_type == 'grid':
            cg_type = GridGroup()
        elif self.group_type == 'thread_block':
            cg_type = ThreadBlockGroup()
        else:
            raise NotImplementedError
        return Data(f'cg::this_{self.group_type}()', cg_type)


this_grid = ThisCgGroup('grid')
this_thread_block = ThisCgGroup('thread_block')
