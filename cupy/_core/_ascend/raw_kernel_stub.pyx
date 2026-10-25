import pickle
import warnings

from cupy import _util

from cupy.backends.backend.api cimport runtime
from cupy.xpu.function cimport Function, Module

from cupy.backends.ascend import bisheng


cdef class RawKernel:
    """User-defined custom AscendC kernel.

    This class is the Ascend counterpart of CuPy's ``RawKernel``: the source
    is `AscendC <https://www.hiascend.com/document>`_ instead of CUDA C++,
    and the compiler is the bisheng device compiler from the CANN toolkit
    instead of NVRTC. The kernel is compiled at the first invocation of the
    :meth:`~RawKernel.__call__` method (or :meth:`compile`) and cached for
    each device; the compiled fatbin is also cached on disk under
    ``~/.cache/cupy/cann_kernel_cache/`` (override the base with
    ``CUPY_CACHE_DIR``) with a hash file name, so it is reused across
    processes.

    Args:
        code (str): AscendC source code. It must define ``extern "C"``
            ``__global__ __aicore__`` kernel functions; the CANN AscendC
            include directories (``kernel_operator.h`` ...) are added
            automatically.
        name (str): Name of the kernel function (the entry point looked up
            via ``aclrtBinaryGetFunction``).
        options (tuple of str): Extra bisheng command line options, e.g.
            ``('--O2', '--cce-soc-version=Ascend910B4',)``. Part of the
            cache key.
        backend (str): Accepted for signature compatibility; on Ascend the
            only compiler is ``'bisheng'``. Any other value warns and is
            ignored.

    .. note::
        **grid/block mapping.** Ascend has no grid x block hierarchy: one
        launch starts ``numBlocks`` kernel instances and each instance sees
        ``GetBlockIdx()``/``GetBlockNum()``. ``RawKernel`` uses the confirmed
        product mapping (docs/ascend/CustomKernel.md §3.4.1, decision #3):

        ``numBlocks = grid[0]*grid[1]*grid[2] * block[0]*block[1]*block[2]``

        so a kernel instance corresponds to one CUDA *thread*:

        .. code-block:: c++

            uint64_t tid = GetBlockIdx();       // [0, G*B)
            uint64_t block = tid / B;           // CUDA blockIdx
            uint64_t thread = tid % B;          // CUDA threadIdx

        For AscendC-native kernels (recommended: each instance processes a
        contiguous vector chunk) simply pass ``block=(1, 1, 1)`` — the
        mapping then degenerates to ``numBlocks = grid``.

    .. note::
        As on CUDA, views of CuPy arrays are ignored and strides must be
        handled manually in the kernel; kernels are compiled out-of-tree
        (no device-code linking: ``LinkState`` is unsupported on Ascend).
    """
    def __cinit__(self):
        # this is only for pickling: if any change is made such that the old
        # pickles cannot be reused, we bump this version number
        self.raw_ver = 2

    def __init__(self, str code, str name, tuple options=(),
                 str backend='bisheng', *, bint translate_cucomplex=False,
                 bint enable_cooperative_groups=False, object jitify=None):
        if translate_cucomplex:
            raise ValueError(
                'translate_cucomplex is not supported on Ascend '
                '(there is no cuComplex.h to translate)')
        if enable_cooperative_groups:
            raise ValueError(
                'cooperative groups are not supported on Ascend')
        if jitify is not None:
            warnings.warn(
                'the jitify argument is a no-op on Ascend '
                '(no Jitify equivalent for bisheng)', RuntimeWarning,
                stacklevel=3)
        if backend != 'bisheng':
            warnings.warn(
                f'backend={backend!r} is not available on Ascend; '
                'the bisheng compiler will be used', RuntimeWarning,
                stacklevel=3)

        self.code = code
        self.name = name
        self.options = options
        self.backend = backend
        self.translate_cucomplex = translate_cucomplex
        self.enable_cooperative_groups = enable_cooperative_groups
        self.jitify = jitify

        # only used when RawKernels are produced from RawModule
        self.file_path = None  # for prebuilt fatbins
        self.name_expressions = None  # no C++ template lowering on Ascend

        # per-device, per-instance cache, to be initialized on first call
        self._kernel_cache = []

        # This is for profiling mechanisms to auto infer a name
        self.__name__ = name

    def __call__(self, grid, block, args, **kwargs):
        """__call__(self, grid, block, args, \\*, shared_mem=0, stream=None)

        Compiles and invokes the kernel.

        The compilation (bisheng) runs only if the kernel is not cached
        (in-process per device, and on disk across processes).

        Args:
            grid (tuple): Size of grid (mapped to ``numBlocks``, see the
                class docstring for the grid/block product mapping).
            block (tuple): Dimensions of each thread block. Folded into
                ``numBlocks``; pass ``(1, 1, 1)`` for AscendC-native kernels.
            args (tuple): Arguments of the kernel. Same packing rules as
                CUDA: ``cupy.ndarray`` -> device pointer, scalar -> by
                value, ``None`` -> NULL.
            shared_mem (int): Dynamic unified-buffer (UB) size per block in
                bytes; requires CANN 9.0+.
            stream (cupy.cuda.Stream): Stream to run on (default: current).
        """
        grid = (tuple(grid) + (1, 1, 1))[:3]
        block = (tuple(block) + (1, 1, 1))[:3]
        num_blocks = 1
        for v in grid:
            if v > 0:
                num_blocks *= v
        for v in block:
            if v > 0:
                num_blocks *= v
        # The product mapping is collapsed here: the Function layer is
        # intentionally thin ("launch exactly the numBlocks given"), so we
        # feed it a one-dimensional grid of G*B instances.
        self._kernel()((num_blocks, 1, 1), (1, 1, 1), args, **kwargs)

    @property
    def kernel(self):
        return self._kernel()

    def _kernel(self, log_stream=None):
        # The kernel is cached, so on the device where this has been called,
        # we would just look up from the cache, and do recompiling only when
        # switching to a different device
        cdef Function ker
        cdef Module mod

        # We delay establishing the device context until it's really needed
        cdef int dev = runtime.getDevice()
        if not self._kernel_cache:
            self._kernel_cache = [None] * runtime.getDeviceCount()

        ker = self._kernel_cache[dev]
        if ker is None:
            assert (self.code is None) != (self.file_path is None)
            mod = _get_raw_module(
                self.code, self.file_path, self.options, log_stream)
            ker = mod.get_function(self.name)
            self._kernel_cache[dev] = ker
        return ker

    # Pickling: cdef classes cannot define __reduce__ with keyword-only
    # arguments, so (as upstream) we go through __getstate__/__setstate__.

    def __getstate__(self):
        cdef dict args
        args = {'code': self.code,
                'name': self.name,
                'options': self.options,
                'backend': self.backend,
                'translate_cucomplex': self.translate_cucomplex,
                'file_path': self.file_path,
                'name_expressions': self.name_expressions,
                'enable_cooperative_groups': self.enable_cooperative_groups,
                'jitify': self.jitify,
                'raw_ver': self.raw_ver}
        return args

    def __setstate__(self, dict args):
        if args.get('raw_ver') != self.raw_ver:
            raise pickle.UnpicklingError(
                'The pickled RawKernel object is not supported by the '
                'current version. It should not be used. Please recompile.')

        self.code = args['code']
        self.name = self.__name__ = args['name']
        self.options = args['options']
        self.backend = args['backend']
        self.translate_cucomplex = args['translate_cucomplex']
        self.enable_cooperative_groups = args['enable_cooperative_groups']
        self.file_path = args['file_path']
        self.name_expressions = args['name_expressions']
        self.jitify = args['jitify']
        self._kernel_cache = []  # to force recompiling

    @property
    def attributes(self):
        """Not available on Ascend: there is no aclrt equivalent of
        ``cuFuncGetAttribute``."""
        raise NotImplementedError(
            'kernel attributes are not supported on Ascend '
            '(no aclrt equivalent of cuFuncGetAttribute)')

    def compile(self, log_stream=None):
        """Compile the current kernel.

        In general, you don't have to call this method;
        kernels are compiled implicitly on the first call.

        Args:
            log_stream (object): Pass either ``sys.stdout`` or a file object
                to which the compiler output will be written.
                Defaults to ``None``.
        """
        # Flush the cache when compilation is explicitly requested
        self._kernel_cache = [None] * runtime.getDeviceCount()
        self._kernel(log_stream=log_stream)


cdef class RawModule:
    """User-defined custom AscendC module.

    This is the Ascend counterpart of CuPy's ``RawModule``: it either
    compiles AscendC source with the bisheng compiler (fatbin cached on
    disk, see :class:`RawKernel`) or loads a prebuilt fatbin (``.o``
    compiled by bisheng for the running SoC) from a file path.

    Kernels are retrieved via :meth:`get_function`, which returns a
    :class:`RawKernel`.

    Args:
        code (str): AscendC source code. Mutually exclusive with ``path``.
        path (str): Path to a bisheng-compiled fatbin (``.o``). Mutually
            exclusive with ``code``.
        options (tuple of str): Extra bisheng command line options.
        backend (str): Accepted for signature compatibility; on Ascend the
            only compiler is ``'bisheng'``.
        name_expressions (sequence of str): Not supported on Ascend
            (AscendC kernels are ``extern "C"``; bisheng has no equivalent
            of NVRTC's lowered-name lookup for C++ templates).
    """
    def __init__(self, *, str code=None, str path=None, tuple options=(),
                 str backend='bisheng', bint translate_cucomplex=False,
                 bint enable_cooperative_groups=False,
                 name_expressions=None, jitify=None):
        if (code is None) == (path is None):
            raise TypeError(
                'Exactly one of `code` and `path` keyword arguments must be '
                'given.')
        # Deferred check: upstream semantics compile lazily, and cupyx.scipy
        # sparse builds RawModule(name_expressions=...) at import time; the
        # error must surface only when the module is actually used.
        if name_expressions is not None:
            name_expressions = tuple(name_expressions)  # make it hashable
        if translate_cucomplex:
            raise ValueError(
                'translate_cucomplex is not supported on Ascend '
                '(there is no cuComplex.h to translate)')
        if enable_cooperative_groups:
            raise ValueError(
                'cooperative groups are not supported on Ascend')
        if jitify is not None:
            warnings.warn(
                'the jitify argument is a no-op on Ascend '
                '(no Jitify equivalent for bisheng)', RuntimeWarning,
                stacklevel=3)
        if backend != 'bisheng':
            warnings.warn(
                f'backend={backend!r} is not available on Ascend; '
                'the bisheng compiler will be used', RuntimeWarning,
                stacklevel=3)

        self.code = code
        self.file_path = path
        self.options = options
        self.backend = backend
        self.enable_cooperative_groups = enable_cooperative_groups
        self.jitify = jitify
        self.translate_cucomplex = translate_cucomplex
        self.name_expressions = name_expressions

    @property
    def module(self):
        return self._module()

    def _module(self, log_stream=None):
        # The module is cached (memoized per device), so on the device where
        # this has been called, we would just look up from the cache, and do
        # recompiling only when switching to a different device
        cdef Module mod

        if self.code is not None and self.name_expressions is not None:
            raise NotImplementedError(
                'name_expressions is not supported on Ascend: AscendC '
                'kernels are extern "C" and bisheng provides no equivalent '
                'of NVRTC name lowering for C++ template kernels')

        mod = _get_raw_module(
            self.code, self.file_path, self.options, log_stream)
        return mod

    def compile(self, log_stream=None):
        """Compile the current module.

        In general, you don't have to call this method;
        kernels are compiled implicitly on the first call.

        Args:
            log_stream (object): Pass either ``sys.stdout`` or a file object
                to which the compiler output will be written.
                Defaults to ``None``.
        """
        self._module(log_stream)

    def get_function(self, str name):
        """Retrieve an AscendC kernel by its entry-point name.

        Args:
            name (str): Name of the ``extern "C" __global__ __aicore__``
                kernel function.

        Returns:
            RawKernel: A :class:`RawKernel` instance.
        """
        cdef RawKernel ker

        if self.name_expressions is not None:
            raise NotImplementedError(
                'name_expressions is not supported on Ascend: AscendC '
                'kernels are extern "C" and bisheng provides no equivalent '
                'of NVRTC name lowering for C++ template kernels')

        ker = RawKernel(
            self.code, name, self.options, self.backend)

        # for lookup in case we loaded from a prebuilt fatbin
        ker.file_path = self.file_path
        # for lookup in case we specialize a template
        ker.name_expressions = self.name_expressions
        # register the kernel in the cache
        func = ker.kernel  # no-cython-lint
        return ker

    def get_global(self, name):
        '''Retrieve a pointer to a global symbol by its name from the module.

        Not available on Ascend: there is no aclrt equivalent of
        ``cuModuleGetGlobal``.
        '''
        raise NotImplementedError(
            'module global variables are not supported on Ascend '
            '(no aclrt equivalent of cuModuleGetGlobal)')


@_util.memoize(for_each_device=True)
def _get_raw_module(str code, str path, tuple options, object log_stream):
    cdef Module mod
    mod = Module()
    if code is not None:
        fatbin = bisheng.compile_module_with_cache(
            code, options, log_stream=log_stream)
        mod.load(fatbin)
    elif path is not None:
        mod.load_file(path)
    return mod
