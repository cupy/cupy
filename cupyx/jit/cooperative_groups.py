from cupyx.jit._cuda_types import TypeBase, void
from cupyx.jit._internal_types import BuiltinFunc
from cupyx.jit._internal_types import Data


# public interface of this module
__all__ = ['this_grid', 'this_thread_block']


class ThreadGroup(TypeBase):
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
        return Data('sync()', void)


class GridGroup(ThreadGroup):

    def __init__(self):
        self.child_type = 'cg::grid_group'

    def is_valid(self):
        self._check_cg_include(env)
        raise NotImplementedError

    def sync(self, env):
        self._check_cg_include(env)
        env.generated.enable_cg = True
        return super().sync(env)


class ThreadBlockGroup(ThreadGroup):

    def __init__(self):
        self.child_type = 'cg::thread_block'


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
