import warnings

import cupy
from cupy import util
from cupy.cuda cimport driver
from cupy.cuda.function cimport Module


cdef class RawKernel:

    """User-defined custom kernel.

    This class can be used to define a custom kernel using raw CUDA source.

    The kernel is compiled at an invocation of the :meth:`~RawKernel.__call__`
    method, which is cached for each device.
    The compiled binary is also cached into a file under the
    ``$HOME/.cupy/kernel_cache/`` directory with a hashed file name. The cached
    binary is reused by other processes.

    Args:
        code (str): CUDA source code.
        name (str): Name of the kernel function.
        options (tuple of str): Compiler options passed to the backend (NVRTC
            or NVCC). For details, see
            https://docs.nvidia.com/cuda/nvrtc/index.html#group__options or
            https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#command-option-description
        backend (str): Either `nvrtc` or `nvcc`. Defaults to `nvrtc`
        translate_cucomplex (bool): Whether the CUDA source includes the header
            `cuComplex.h` or not. If set to ``True``, any code that uses the
            functions from `cuComplex.h` will be translated to its Thrust
            counterpart. Defaults to ``False``.
        enable_cooperative_groups (bool): Whether to enable cooperative groups
            in the CUDA source. If set to ``True``, compile options are
            configured properly and the kernel is launched with
            ``cuLaunchCooperativeKernel`` so that cooperative groups can be
            used from the CUDA source.
            This feature is only supported in CUDA 9 or later.
    """

    def __init__(self, code, name, options=(), backend='nvrtc', *,
                 translate_cucomplex=False, enable_cooperative_groups=False):
        if isinstance(code, bytes):
            code = code.decode('UTF-8')
        if isinstance(name, bytes):
            name = name.decode('UTF-8')
        if isinstance(backend, bytes):
            backend = backend.decode('UTF-8')

        self.code = code
        self.name = name
        self.options = options
        self.backend = backend
        self.translate_cucomplex = translate_cucomplex
        self.enable_cooperative_groups = enable_cooperative_groups
        self._kernel = None

    def __call__(self, grid, block, args, **kwargs):
        """__call__(self, grid, block, args, *, shared_mem=0)

        Compiles and invokes the kernel.

        The compilation runs only if the kernel is not cached.

        Args:
            grid (tuple): Size of grid in blocks.
            block (tuple): Dimensions of each thread block.
            args (tuple): Arguments of the kernel.
            shared_mem (int): Dynamic shared-memory size per thread block in
                bytes.

        """
        self.kernel(
            grid, block, args,
            **kwargs)

    @property
    def kernel(self):
        if self._kernel is None:
            self._kernel = _get_raw_kernel(
                self.code, self.name, self.options, self.backend,
                self.translate_cucomplex, self.enable_cooperative_groups)
        return self._kernel

    @property
    def attributes(self):
        """ (Deprecated) use `RawKernel.function.attributes` instead. """
        warnings.warn(
            'RawKernel.attributes is deprecated. '
            'Use RawKernel.function.attributes instead.',
            DeprecationWarning)
        return self.function.attributes()

    @property
    def max_threads_per_block(self):
        """ (Deprecated) use `RawKernel.function.max_threads_per_block`
        instead.
        """
        warnings.warn(
            'RawKernel.max_threads_per_block is deprecated. '
            'Use RawKernel.function.max_threads_per_block instead.',
            DeprecationWarning)
        return self.function.max_threads_per_block()

    @property
    def shared_size_bytes(self):
        """ (Deprecated) use `RawKernel.function.shared_size_bytes`
        instead.
        """
        warnings.warn(
            'RawKernel.shared_size_bytes is deprecated. '
            'Use RawKernel.function.shared_size_bytes instead.',
            DeprecationWarning)
        return self.function.shared_size_bytes()

    @property
    def const_size_bytes(self):
        """ (Deprecated) use `RawKernel.function.const_size_bytes`
        instead.
        """
        warnings.warn(
            'RawKernel.const_size_bytes is deprecated. '
            'Use RawKernel.function.const_size_bytes instead.',
            DeprecationWarning)
        return self.function.const_size_bytes()

    @property
    def local_size_bytes(self):
        """ (Deprecated) use `RawKernel.function.local_size_bytes`
        instead.
        """
        warnings.warn(
            'RawKernel.local_size_bytes is deprecated. '
            'Use RawKernel.function.local_size_bytes instead.',
            DeprecationWarning)
        return self.function.local_size_bytes()

    @property
    def num_regs(self):
        """ (Deprecated) use `RawKernel.function.num_regs`
        instead.
        """
        warnings.warn(
            'RawKernel.num_regs is deprecated. '
            'Use RawKernel.function.num_regs instead.',
            DeprecationWarning)
        return self.function.num_regs()

    @property
    def ptx_version(self):
        """ (Deprecated) use `RawKernel.function.ptx_version`
        instead.
        """
        warnings.warn(
            'RawKernel.ptx_version is deprecated. '
            'Use RawKernel.function.ptx_version instead.',
            DeprecationWarning)
        return self.function.ptx_version()

    @property
    def binary_version(self):
        """ (Deprecated) use `RawKernel.function.binary_version`
        instead.
        """
        warnings.warn(
            'RawKernel.binary_version is deprecated. '
            'Use RawKernel.function.binary_version instead.',
            DeprecationWarning)
        return self.function.binary_version()

    @property
    def cache_mode_ca(self):
        """ (Deprecated) use `RawKernel.function.cache_mode_ca`
        instead.
        """
        warnings.warn(
            'RawKernel.cache_mode_ca is deprecated. '
            'Use RawKernel.function.cache_mode_ca instead.',
            DeprecationWarning)
        return self.function.cache_mode_ca()

    @property
    def max_dynamic_shared_size_bytes(self):
        """ (Deprecated) use `RawKernel.function.max_dynamic_shared_size_bytes`
        instead.
        """
        warnings.warn(
            'RawKernel.max_dynamic_shared_size_bytes is deprecated. '
            'Use RawKernel.function.max_dynamic_shared_size_bytes instead.',
            DeprecationWarning)
        return self.function.max_dynamic_shared_size_bytes()

    @max_dynamic_shared_size_bytes.setter
    def max_dynamic_shared_size_bytes(self, bytes):
        """ (Deprecated) use `RawKernel.function.max_dynamic_shared_size_bytes`
        instead.
        """
        warnings.warn(
            'RawKernel.max_dynamic_shared_size_bytes is deprecated. '
            'Use RawKernel.function.max_dynamic_shared_size_bytes instead.',
            DeprecationWarning)
        self.function.max_dynamic_shared_size_bytes(bytes)

    @property
    def preferred_shared_memory_carveout(self):
        """ (Deprecated) use `RawKernel.function.preferred_shared_memory_carveout`
        instead.
        """
        warnings.warn(
            'RawKernel.preferred_shared_memory_carveout is deprecated. '
            'Use RawKernel.function.preferred_shared_memory_carveout instead.',
            DeprecationWarning)
        return self.function.preferred_shared_memory_carveout()

    @preferred_shared_memory_carveout.setter
    def preferred_shared_memory_carveout(self, fraction):
        """ (Deprecated) use `RawKernel.function.preferred_shared_memory_carveout`
        instead.
        """
        warnings.warn(
            'RawKernel.preferred_shared_memory_carveout is deprecated. '
            'Use RawKernel.function.preferred_shared_memory_carveout instead.',
            DeprecationWarning)
        self.kernel.preferred_shared_memory_carveout(fraction)


@cupy.util.memoize(for_each_device=True)
def _get_raw_kernel(code, name, options=(), backend='nvrtc',
                    translate_cucomplex=False,
                    enable_cooperative_groups=False):
    module = cupy.core.core.compile_with_cache(
        code, options, prepend_cupy_headers=False, backend=backend,
        translate_cucomplex=translate_cucomplex,
        enable_cooperative_groups=enable_cooperative_groups)
    return module.get_function(name)


cdef class RawModule:
    """User-defined custom module.

    This class can be used to either compile raw CUDA sources or load CUDA
    modules (\\*.cubin, \\*.ptx). This class is useful when a number of CUDA
    kernels in the same source need to be retrieved.

    For the former case, the CUDA source code is compiled when initializing a
    new instance of this class, and the kernels can be retrieved by calling
    :meth:`get_function`, which will return an instance of :class:`RawKernel`.
    (Same as in :class:`RawKernel`, the generated binary is also cached.)

    For the latter case, an existing CUDA binary (\\*.cubin) or a PTX file can
    be loaded by providing its path, and kernels therein can be retrieved
    similarly.

    Args:
        code (str): CUDA source code. Mutually exclusive with ``path``.
        path (str): Path to cubin/ptx. Mutually exclusive with ``code``.
        options (tuple of str): Compiler options passed to the backend (NVRTC
            or NVCC). For details, see
            https://docs.nvidia.com/cuda/nvrtc/index.html#group__options or
            https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#command-option-description
        backend (str): Either `nvrtc` or `nvcc`. Defaults to `nvrtc`
        translate_cucomplex (bool): Whether the CUDA source includes the header
            `cuComplex.h` or not. If set to ``True``, any code that uses the
            functions from `cuComplex.h` will be translated to its Thrust
            counterpart. Defaults to ``False``.
        enable_cooperative_groups (bool): Whether to enable cooperative groups
            in the CUDA source. If set to ``True``, compile options are
            configured properly and the kernel is launched with
            ``cuLaunchCooperativeKernel`` so that cooperative groups can be
            used from the CUDA source.
            This feature is only supported in CUDA 9 or later.

    .. note::
        Each kernel in ``RawModule`` possesses independent function attributes.
    """
    def __init__(self, *, code=None, path=None, options=(), backend='nvrtc',
                 translate_cucomplex=False, enable_cooperative_groups=False):
        if (code is None) == (path is None):
            raise TypeError(
                'Exactly one of `code` and `path` keyword arguments must be '
                'given.')
        if path is not None and isinstance(path, bytes):
            path = path.decode('UTF-8')
        if code is not None and isinstance(code, bytes):
            code = code.decode('UTF-8')
        if isinstance(backend, bytes):
            backend = backend.decode('UTF-8')

        self.code = code
        self.cubin_path = path
        self.enable_cooperative_groups = enable_cooperative_groups

        if self.code is not None:
            self.module = cupy.core.core.compile_with_cache(
                code, options, prepend_cupy_headers=False, backend=backend,
                translate_cucomplex=translate_cucomplex,
                enable_cooperative_groups=self.enable_cooperative_groups)
            self.options = options
            self.backend = backend
            self.translate_cucomplex = translate_cucomplex
        elif self.cubin_path is not None:
            self.module = Module(enable_cooperative_groups)
            self.module.load_file(self.cubin_path)
            self.options = ()
            self.backend = 'nvcc'
            self.translate_cucomplex = False

        self.kernels = {}

    def get_function(self, name):
        """Retrieve a CUDA kernel by its name from the module.

        Args:
            name (str): Name of the kernel function.

        Returns:
            RawKernel: An ``RawKernel`` instance.
        """
        if name in self.kernels:
            return self.kernels[name]
        else:
            kernel = self.module.get_function(name)
            self.kernels[name] = kernel
            return kernel

    def get_texref(self, name):
        '''Retrieve a texture reference by its name from the module.

        Args:
            name (str): Name of the texture reference.

        Returns:
            intptr_t: A ``CUtexref`` handle, to be passed to :class:`~cupy.cuda.texture.TextureReference`.
        '''  # noqa
        return self.module.get_texref(name)
