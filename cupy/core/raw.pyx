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
        code (str): CUDA source code. Mutually exclusive with ``kernel``
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
        kernel (:class:`cupy.cuda.Function`): CUDA Kernel object 
            to be executed. Mutually exclusive with ``code``
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
        self._kernel = kernel

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
            self.module = Module()
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
