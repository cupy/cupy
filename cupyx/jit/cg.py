from cupy.cuda import runtime as _runtime
from cupyx.jit import _compile
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import BuiltinFunc as _BuiltinFunc
from cupyx.jit._internal_types import Constant as _Constant
from cupyx.jit._internal_types import Data as _Data


# public interface of this module
__all__ = ['this_grid', 'this_thread_block',
           'sync', 'wait', 'wait_prior', 'memcpy_async']


def _check_cg_include(env, target='cg'):
    if target == 'cg':
        included = env.generated.include_cg
        if included is False:
            # prepend the header
            env.generated.codes.insert(
                0, "\n#include <cooperative_groups.h>")
            env.generated.codes.insert(
                1, "namespace cg = cooperative_groups;\n")
            env.generated.include_cg = True
    elif target == 'memcpy_async':
        _check_cg_include(env)
        included = env.generated.include_cg_memcpy_async
        if included is False:
            # prepend the header
            env.generated.codes.insert(
                1, "#include <cooperative_groups/memcpy_async.h>")
            env.generated.include_cg_memcpy_async = True
    elif target == 'cuda_barrier':
        included = env.generated.include_cuda_barrier
        if included is False:
            # prepend the header
            env.generated.codes.insert(
                1, "#include <cuda/barrier>")
            env.generated.include_cuda_barrier = True


class _ThreadGroup(_cuda_types.TypeBase):
    """ Base class for all cooperative groups. """

    child_type = None

    def __init__(self):
        raise NotImplementedError

    def __str__(self):
        return f'{self.child_type}'

    def sync(self, env):
        _check_cg_include(env)
        return _Data('sync()', _cuda_types.void)


class _GridGroup(_ThreadGroup):
    """A handle to the current grid group. Must be created via :func:`this_grid`.

    .. seealso:: `CUDA Grid Group API`_, :class:`numba.cuda.cg.GridGroup`

    .. _CUDA Grid Group API:
        https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#grid-group-cg
    """

    def __init__(self):
        self.child_type = 'cg::grid_group'

    def is_valid(self, env):
        """
        is_valid()

        Returns whether the grid_group can synchronize.
        """
        _check_cg_include(env)
        return _Data('is_valid()', _cuda_types.bool_)

    def sync(self, env):
        """
        sync()

        Synchronize the threads named in the group.

        .. seealso:: :meth:`numba.cuda.cg.GridGroup.sync`
        """
        # when this methond is called, we need to use the cooperative
        # launch API
        env.generated.enable_cg = True
        return super().sync(env)

    def thread_rank(self, env):
        """
        thread_rank()

        Rank of the calling thread within ``[0, num_threads)``.
        """
        _check_cg_include(env)
        return _Data('thread_rank()', _cuda_types.uint64)

    def block_rank(self, env):
        """
        block_rank()

        Rank of the calling block within ``[0, num_blocks)``.
        """
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("block_rank() is supported on CUDA 11.6+")
        _check_cg_include(env)
        return _Data('block_rank()', _cuda_types.uint64)

    def num_threads(self, env):
        """
        num_threads()

        Total number of threads in the group.
        """
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("num_threads() is supported on CUDA 11.6+")
        _check_cg_include(env)
        return _Data('num_threads()', _cuda_types.uint64)

    def num_blocks(self, env):
        """
        num_blocks()

        Total number of blocks in the group.
        """
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("num_blocks() is supported on CUDA 11.6+")
        _check_cg_include(env)
        return _Data('num_blocks()', _cuda_types.uint64)

    def dim_blocks(self, env):
        """
        dim_blocks()

        Dimensions of the launched grid in units of blocks.
        """
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("dim_blocks() is supported on CUDA 11.6+")
        from cupyx.jit._interface import _Dim3  # avoid circular import
        _check_cg_include(env)
        return _Data('dim_blocks()', _Dim3())

    def block_index(self, env):
        """
        block_index()

        3-Dimensional index of the block within the launched grid.
        """
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("block_index() is supported on CUDA 11.6+")
        from cupyx.jit._interface import _Dim3  # avoid circular import
        _check_cg_include(env)
        return _Data('block_index()', _Dim3())

    def size(self, env):
        """
        size()

        Total number of threads in the group.
        """
        # despite it is an alias of num_threads, we need it for earlier 11.x
        _check_cg_include(env)
        return _Data('size()', _cuda_types.uint64)

    def group_dim(self, env):
        """
        group_dim()

        Dimensions of the launched grid in units of blocks.
        """
        # despite it is an alias of dim_blocks, we need it for earlier 11.x
        from cupyx.jit._interface import _Dim3  # avoid circular import
        _check_cg_include(env)
        return _Data('group_dim()', _Dim3())


class _ThreadBlockGroup(_ThreadGroup):
    """A handle to the current thread block group. Must be
    created via :func:`this_thread_block`.

    .. seealso:: `CUDA Thread Block Group API`_

    .. _CUDA Thread Block Group API:
        https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-group-cg
    """

    def __init__(self):
        self.child_type = 'cg::thread_block'

    def sync(self, env):
        """
        sync()

        Synchronize the threads named in the group.
        """
        return super().sync(env)

    def thread_rank(self, env):
        """
        thread_rank()

        Rank of the calling thread within ``[0, num_threads)``.
        """
        _check_cg_include(env)
        return _Data('thread_rank()', _cuda_types.uint32)

    def group_index(self, env):
        """
        group_index()

        3-Dimensional index of the block within the launched grid.
        """
        from cupyx.jit._interface import _Dim3  # avoid circular import
        _check_cg_include(env)
        return _Data('group_index()', _Dim3())

    def thread_index(self, env):
        """
        thread_index()

        3-Dimensional index of the thread within the launched block.
        """
        from cupyx.jit._interface import _Dim3  # avoid circular import
        _check_cg_include(env)
        return _Data('thread_index()', _Dim3())

    def dim_threads(self, env):
        """
        dim_threads()

        Dimensions of the launched block in units of threads.
        """
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("dim_threads() is supported on CUDA 11.6+")
        from cupyx.jit._interface import _Dim3  # avoid circular import
        _check_cg_include(env)
        return _Data('dim_threads()', _Dim3())

    def num_threads(self, env):
        """
        num_threads()

        Total number of threads in the group.
        """
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("num_threads() is supported on CUDA 11.6+")
        _check_cg_include(env)
        return _Data('num_threads()', _cuda_types.uint32)

    def size(self, env):
        """
        size()

        Total number of threads in the group.
        """
        # despite it is an alias of num_threads, we need it for earlier 11.x
        _check_cg_include(env)
        return _Data('size()', _cuda_types.uint32)

    def group_dim(self, env):
        """
        group_dim()

        Dimensions of the launched block in units of threads.
        """
        # despite it is an alias of dim_threads, we need it for earlier 11.x
        from cupyx.jit._interface import _Dim3  # avoid circular import
        _check_cg_include(env)
        return _Data('group_dim()', _Dim3())


class _ThisCgGroup(_BuiltinFunc):

    def __init__(self, group_type):
        if group_type == "grid":
            name = "grid group"
            typename = "_GridGroup"
        elif group_type == 'thread_block':
            name = "thread block group"
            typename = "_ThreadBlockGroup"
        else:
            raise NotImplementedError
        self.group_type = group_type
        self.__doc__ = f"""Get the current {name}.

        .. seealso:: :class:`cupyx.jit.cg.{typename}`"""
        if group_type == "grid":
            self.__doc__ += ", :func:`numba.cuda.cg.this_grid`"

    def call_const(self, env):
        if _runtime.is_hip:
            raise RuntimeError('cooperative group is not supported on HIP')
        if self.group_type == 'grid':
            if _runtime.runtimeGetVersion() < 11000:
                raise RuntimeError(
                    "For pre-CUDA 11, the grid group has very limited "
                    "functionality (only group.sync() works), and so we "
                    "disable the grid group support to prepare the transition "
                    "to support CUDA 11+ only.")
            cg_type = _GridGroup()
        elif self.group_type == 'thread_block':
            cg_type = _ThreadBlockGroup()
        return _Data(f'cg::this_{self.group_type}()', cg_type)


class _Sync(_BuiltinFunc):

    def __call__(self):
        """Calls ``cg::sync()``.

        """
        super().__call__()

    def call(self, env, group):
        _check_cg_include(env)
        return _Data(f'cg::sync({group.code})', _cuda_types.void)


class _MemcpySync(_BuiltinFunc):

    def __call__(self):
        """Calls ``cg::memcpy_sync()``.

        """
        super().__call__()

    def call(self, env, group, dst, dst_idx, src, src_idx, size, *,
             aligned_size=None):
        _check_cg_include(env, target='memcpy_async')

        dst = _Data.init(dst, env)
        src = _Data.init(src, env)
        for arr in (dst, src):
            if not isinstance(
                    arr.ctype, (_cuda_types.CArray, _cuda_types.Ptr)):
                print(arr, arr.ctype)
                raise TypeError('dst/src must be of array type.')
        dst = _compile._indexing(dst, dst_idx, env)
        src = _compile._indexing(src, src_idx, env)

        size = _compile._astype_scalar(
            # it's very unlikely that the size would exceed 2^32, so we just
            # pick uint32 for simplicity
            size, _cuda_types.uint32, 'same_kind', env)
        size = _Data.init(size, env)
        size_code = f'{size.code}'
        if aligned_size:
            if not isinstance(aligned_size, _Constant):
                raise ValueError(
                    'aligned_size must be a compile-time constant')
            _check_cg_include(env, target='cuda_barrier')
            size_code = (f'cuda::aligned_size_t<{aligned_size.obj}>'
                         f'({size_code})')
        return _Data(f'cg::memcpy_async({group.code}, &({dst.code}), '
                     f'&({src.code}), {size_code})', _cuda_types.void)


class _Wait(_BuiltinFunc):

    def __call__(self):
        """Calls ``cg::wait()``.

        """
        super().__call__()

    def call(self, env, group):
        _check_cg_include(env)
        return _Data(f'cg::wait({group.code})', _cuda_types.void)


class _WaitPrior(_BuiltinFunc):

    def __call__(self):
        """Calls ``cg::wait_prior<N>()``.

        """
        super().__call__()

    def call(self, env, group, step):
        _check_cg_include(env)
        assert isinstance(step, _Constant)
        return _Data(f'cg::wait_prior<{step.obj}>({group.code})',
                     _cuda_types.void)


this_grid = _ThisCgGroup('grid')
this_thread_block = _ThisCgGroup('thread_block')
sync = _Sync()
wait = _Wait()
wait_prior = _WaitPrior()
memcpy_async = _MemcpySync()
