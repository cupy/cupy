from cupy.cuda import runtime as _runtime
from cupyx.jit import _compile
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import BuiltinFunc as _BuiltinFunc
from cupyx.jit._internal_types import Constant as _Constant
from cupyx.jit._internal_types import Data as _Data
from cupyx.jit._internal_types import wraps_class_method as _wraps_class_method


# public interface of this module
__all__ = ['this_grid', 'this_thread_block',
           'sync', 'wait', 'wait_prior', 'memcpy_async']


_header_to_code = {
    'cg': ("#include <cooperative_groups.h>\n"
           "namespace cg = cooperative_groups;\n"),
    'cg_memcpy_async': "#include <cooperative_groups/memcpy_async.h>",
    'cuda_barrier': "#include <cuda/barrier>",
}


def _check_include(env, header):
    flag = getattr(env.generated, f"include_{header}")
    if flag is False:
        # prepend the header
        env.generated.codes.append(_header_to_code[header])
        setattr(env.generated, f"include_{header}", True)


class _ThreadGroup(_cuda_types.TypeBase):
    """ Base class for all cooperative groups. """

    child_type = None

    def __init__(self):
        raise NotImplementedError

    def __str__(self):
        return f'{self.child_type}'

    def _sync(self, env, instance):
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.sync()', _cuda_types.void)


class _GridGroup(_ThreadGroup):
    """A handle to the current grid group. Must be created via :func:`this_grid`.

    .. seealso:: `CUDA Grid Group API`_, :class:`numba.cuda.cg.GridGroup`

    .. _CUDA Grid Group API:
        https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#grid-group-cg
    """

    def __init__(self):
        self.child_type = 'cg::grid_group'

    @_wraps_class_method
    def is_valid(self, env, instance):
        """
        is_valid()

        Returns whether the grid_group can synchronize.
        """
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.is_valid()', _cuda_types.bool_)

    @_wraps_class_method
    def sync(self, env, instance):
        """
        sync()

        Synchronize the threads named in the group.

        .. seealso:: :meth:`numba.cuda.cg.GridGroup.sync`
        """
        # when this methond is called, we need to use the cooperative
        # launch API
        env.generated.enable_cg = True
        return super()._sync(env, instance)

    @_wraps_class_method
    def thread_rank(self, env, instance):
        """
        thread_rank()

        Rank of the calling thread within ``[0, num_threads)``.
        """
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.thread_rank()', _cuda_types.uint64)

    @_wraps_class_method
    def block_rank(self, env, instance):
        """
        block_rank()

        Rank of the calling block within ``[0, num_blocks)``.
        """
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("block_rank() is supported on CUDA 11.6+")
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.block_rank()', _cuda_types.uint64)

    @_wraps_class_method
    def num_threads(self, env, instance):
        """
        num_threads()

        Total number of threads in the group.
        """
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("num_threads() is supported on CUDA 11.6+")
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.num_threads()', _cuda_types.uint64)

    @_wraps_class_method
    def num_blocks(self, env, instance):
        """
        num_blocks()

        Total number of blocks in the group.
        """
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("num_blocks() is supported on CUDA 11.6+")
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.num_blocks()', _cuda_types.uint64)

    @_wraps_class_method
    def dim_blocks(self, env, instance):
        """
        dim_blocks()

        Dimensions of the launched grid in units of blocks.
        """
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("dim_blocks() is supported on CUDA 11.6+")
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.dim_blocks()', _cuda_types.dim3)

    @_wraps_class_method
    def block_index(self, env, instance):
        """
        block_index()

        3-Dimensional index of the block within the launched grid.
        """
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("block_index() is supported on CUDA 11.6+")
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.block_index()', _cuda_types.dim3)

    @_wraps_class_method
    def size(self, env, instance):
        """
        size()

        Total number of threads in the group.
        """
        # despite it is an alias of num_threads, we need it for earlier 11.x
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.size()', _cuda_types.uint64)

    @_wraps_class_method
    def group_dim(self, env, instance):
        """
        group_dim()

        Dimensions of the launched grid in units of blocks.
        """
        # despite it is an alias of dim_blocks, we need it for earlier 11.x
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.group_dim()', _cuda_types.dim3)


class _ThreadBlockGroup(_ThreadGroup):
    """A handle to the current thread block group. Must be
    created via :func:`this_thread_block`.

    .. seealso:: `CUDA Thread Block Group API`_

    .. _CUDA Thread Block Group API:
        https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-group-cg
    """

    def __init__(self):
        self.child_type = 'cg::thread_block'

    @_wraps_class_method
    def sync(self, env, instance):
        """
        sync()

        Synchronize the threads named in the group.
        """
        return super()._sync(env, instance)

    @_wraps_class_method
    def thread_rank(self, env, instance):
        """
        thread_rank()

        Rank of the calling thread within ``[0, num_threads)``.
        """
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.thread_rank()', _cuda_types.uint32)

    @_wraps_class_method
    def group_index(self, env, instance):
        """
        group_index()

        3-Dimensional index of the block within the launched grid.
        """
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.group_index()', _cuda_types.dim3)

    @_wraps_class_method
    def thread_index(self, env, instance):
        """
        thread_index()

        3-Dimensional index of the thread within the launched block.
        """
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.thread_index()', _cuda_types.dim3)

    @_wraps_class_method
    def dim_threads(self, env, instance):
        """
        dim_threads()

        Dimensions of the launched block in units of threads.
        """
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("dim_threads() is supported on CUDA 11.6+")
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.dim_threads()', _cuda_types.dim3)

    @_wraps_class_method
    def num_threads(self, env, instance):
        """
        num_threads()

        Total number of threads in the group.
        """
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError("num_threads() is supported on CUDA 11.6+")
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.num_threads()', _cuda_types.uint32)

    @_wraps_class_method
    def size(self, env, instance):
        """
        size()

        Total number of threads in the group.
        """
        # despite it is an alias of num_threads, we need it for earlier 11.x
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.size()', _cuda_types.uint32)

    @_wraps_class_method
    def group_dim(self, env, instance):
        """
        group_dim()

        Dimensions of the launched block in units of threads.
        """
        # despite it is an alias of dim_threads, we need it for earlier 11.x
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.group_dim()', _cuda_types.dim3)


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
        self.__doc__ = f"""
        Returns the current {name} (:class:`~cupyx.jit.cg.{typename}`).

        .. seealso:: :class:`cupyx.jit.cg.{typename}`"""
        if group_type == "grid":
            self.__doc__ += ", :func:`numba.cuda.cg.this_grid`"

    def __call__(self):
        super().__call__()

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

    def __call__(self, group):
        """Calls ``cg::sync()``.

        Args:
            group: a valid cooperative group

        .. seealso:: `cg::sync`_

        .. _cg::sync:
            https://docs.nvidia.com/cuda/archive/11.6.0/cuda-c-programming-guide/index.html#collectives-cg-sync
        """
        super().__call__()

    def call(self, env, group):
        if _runtime.runtimeGetVersion() < 11000:
            raise RuntimeError("not supported in CUDA < 11.0")
        if not isinstance(group.ctype, _ThreadGroup):
            raise ValueError("group must be a valid cooperative group")
        _check_include(env, 'cg')
        return _Data(f'cg::sync({group.code})', _cuda_types.void)


class _MemcpySync(_BuiltinFunc):

    def __call__(self, group, dst, dst_idx, src, src_idx, size, *,
                 aligned_size=None):
        """Calls ``cg::memcpy_sync()``.

        Args:
            group: a valid cooperative group
            dst: the destination array that can be viewed as a 1D
                C-contiguous array
            dst_idx: the start index of the destination array element
            src: the source array that can be viewed as a 1D C-contiguous
                array
            src_idx: the start index of the source array element
            size (int): the number of bytes to be copied from
                ``src[src_index]`` to ``dst[dst_idx]``
            aligned_size (int): Use ``cuda::aligned_size_t<N>`` to guarantee
                the compiler that ``src``/``dst`` are at least N-bytes aligned.
                The behavior is undefined if the guarantee is not held.

        .. seealso:: `cg::memcpy_sync`_

        .. _cg::memcpy_sync:
            https://docs.nvidia.com/cuda/archive/11.6.0/cuda-c-programming-guide/index.html#collectives-cg-memcpy-async
        """
        super().__call__()

    def call(self, env, group, dst, dst_idx, src, src_idx, size, *,
             aligned_size=None):
        if _runtime.runtimeGetVersion() < 11010:
            # the overloaded version of memcpy_async that we use does not yet
            # exist in CUDA 11.0
            raise RuntimeError("not supported in CUDA < 11.1")
        _check_include(env, 'cg')
        _check_include(env, 'cg_memcpy_async')

        dst = _Data.init(dst, env)
        src = _Data.init(src, env)
        for arr in (dst, src):
            if not isinstance(
                    arr.ctype, (_cuda_types.CArray, _cuda_types.Ptr)):
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
            _check_include(env, 'cuda_barrier')
            size_code = (f'cuda::aligned_size_t<{aligned_size.obj}>'
                         f'({size_code})')
        return _Data(f'cg::memcpy_async({group.code}, &({dst.code}), '
                     f'&({src.code}), {size_code})', _cuda_types.void)


class _Wait(_BuiltinFunc):

    def __call__(self, group):
        """Calls ``cg::wait()``.

        Args:
            group: a valid cooperative group

        .. seealso: `cg::wait`_

        .. _cg::wait:
            https://docs.nvidia.com/cuda/archive/11.6.0/cuda-c-programming-guide/index.html#collectives-cg-wait
        """
        super().__call__()

    def call(self, env, group):
        if _runtime.runtimeGetVersion() < 11000:
            raise RuntimeError("not supported in CUDA < 11.0")
        _check_include(env, 'cg')
        return _Data(f'cg::wait({group.code})', _cuda_types.void)


class _WaitPrior(_BuiltinFunc):

    def __call__(self, group):
        """Calls ``cg::wait_prior<N>()``.

        Args:
            group: a valid cooperative group
            step (int): wait for the first ``N`` steps to finish

        .. seealso: `cg::wait_prior`_

        .. _cg::wait_prior:
            https://docs.nvidia.com/cuda/archive/11.6.0/cuda-c-programming-guide/index.html#collectives-cg-wait
        """
        super().__call__()

    def call(self, env, group, step):
        if _runtime.runtimeGetVersion() < 11000:
            raise RuntimeError("not supported in CUDA < 11.0")
        _check_include(env, 'cg')
        if not isinstance(step, _Constant):
            raise ValueError('step must be a compile-time constant')
        return _Data(f'cg::wait_prior<{step.obj}>({group.code})',
                     _cuda_types.void)


this_grid = _ThisCgGroup('grid')
this_thread_block = _ThisCgGroup('thread_block')
sync = _Sync()
wait = _Wait()
wait_prior = _WaitPrior()
memcpy_async = _MemcpySync()
