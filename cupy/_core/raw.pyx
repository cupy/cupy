import pickle

import cupy

from cupy_backends.cuda.api cimport driver
from cupy_backends.cuda.api cimport runtime
from cupy.cuda.function cimport Function, Module


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
        jitify (bool): Whether or not to use `Jitify`_ to assist NVRTC to
            compile C++ kernels. Defaults to ``False``.

    .. _Jitify:
        https://github.com/NVIDIA/jitify

    """
    def __cinit__(self):
        # this is only for pickling: if any change is made such that the old
        # pickles cannot be reused, we bump this version number
        self.raw_ver = 2

    def __init__(self, str code, str name, tuple options=(),
                 str backend='nvrtc', *, bint translate_cucomplex=False,
                 bint enable_cooperative_groups=False, bint jitify=False):

        self.code = code
        self.name = name
        self.options = options
        self.backend = backend
        self.translate_cucomplex = translate_cucomplex
        self.enable_cooperative_groups = enable_cooperative_groups
        self.jitify = jitify

        # only used when RawKernels are produced from RawModule
        self.file_path = None  # for cubin/ptx
        self.name_expressions = None  # for C++ template

        # per-device, per-instance cache, to be initialized on first call
        self._kernel_cache = []

        # This is for profiling mechanisms to auto infer a name
        self.__name__ = name

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
            enable_cooperative_groups=self.enable_cooperative_groups,
            **kwargs)

    @property
    def kernel(self):
        return self._kernel()

    def _kernel(self, log_stream=None):
        # The kernel is cached, so on the device where this has been called,
        # we would just look up from the cache, and do recompiling only when
        # switching to a different device
        cdef Function ker
        cdef Module mod

        # We delay establishing the CUDA context until it's really needed
        cdef int dev = runtime.getDevice()
        if not self._kernel_cache:
            self._kernel_cache = [None] * runtime.getDeviceCount()

        ker = self._kernel_cache[dev]
        if ker is None:
            assert (self.code is None) != (self.file_path is None)
            mod = _get_raw_module(
                self.code, self.file_path, self.options, self.backend,
                self.translate_cucomplex, self.enable_cooperative_groups,
                self.jitify, self.name_expressions, log_stream)
            ker = mod.get_function(self.name)
            self._kernel_cache[dev] = ker
        return ker

    # It is not possible to implement __reduce__ for a cdef class. The
    # two-tuple return cannot handle the keyword-only arguments, and
    # the three-tuple return (for updating the object's internal state)
    # does not work either, because cdef classes by default does not have
    # __dict__. Therefore, the only way to handle keyword-only arguments
    # for picking a cdef class is to define the following two special
    # functions, which is in fact preferred over __reduce__.

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
                'The pickled RawKernel object is not supported by the current '
                'CuPy version. It should not be used. Please recompile.')

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
        """Returns a dictionary containing runtime kernel attributes. This is
        a read-only property; to overwrite the attributes, use

        .. code-block:: python

            kernel = RawKernel(...)  # arguments omitted
            kernel.max_dynamic_shared_size_bytes = ...
            kernel.preferred_shared_memory_carveout = ...

        Note that the two attributes shown in the above example are the only
        two currently settable in CUDA.

        Any attribute not existing in the present CUDA toolkit version will
        have the value -1.

        Returns:
            dict: A dictionary containing the kernel's attributes.
        """
        cdef dict attrs = {}
        cdef list keys = ['max_threads_per_block', 'shared_size_bytes',
                          'const_size_bytes', 'local_size_bytes',
                          'num_regs', 'ptx_version', 'binary_version',
                          'cache_mode_ca', 'max_dynamic_shared_size_bytes',
                          'preferred_shared_memory_carveout']
        for attr in keys:
            attrs[attr] = getattr(self, attr)
        return attrs

    @property
    def max_threads_per_block(self):
        """The maximum number of threads per block that can successfully
        launch the function on the device.
        """
        attr = driver.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @property
    def shared_size_bytes(self):
        """The size in bytes of the statically-allocated shared memory
        used by the function. This is separate from any dynamically-allocated
        shared memory, which must be specified when the function is called.
        """
        attr = driver.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @property
    def const_size_bytes(self):
        """The size in bytes of constant memory used by the function."""
        attr = driver.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @property
    def local_size_bytes(self):
        """The size in bytes of local memory used by the function."""
        attr = driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @property
    def num_regs(self):
        """The number of registers used by the function."""
        attr = driver.CU_FUNC_ATTRIBUTE_NUM_REGS
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @property
    def ptx_version(self):
        """The PTX virtual architecture version that was used during
        compilation, in the format: 10*major + minor.
        """
        attr = driver.CU_FUNC_ATTRIBUTE_PTX_VERSION
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @property
    def binary_version(self):
        """The binary architecture version that was used during compilation,
        in the format: 10*major + minor.
        """
        attr = driver.CU_FUNC_ATTRIBUTE_BINARY_VERSION
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @property
    def cache_mode_ca(self):
        """Indicates whether option "-Xptxas --dlcm=ca" was set during
        compilation.
        """
        attr = driver.CU_FUNC_ATTRIBUTE_CACHE_MODE_CA
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @property
    def max_dynamic_shared_size_bytes(self):
        """The maximum dynamically-allocated shared memory size in bytes that
        can be used by the function. Can be set.
        """
        attr = driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @max_dynamic_shared_size_bytes.setter
    def max_dynamic_shared_size_bytes(self, bytes):
        attr = driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        driver.funcSetAttribute(self.kernel.ptr, attr, bytes)

    @property
    def preferred_shared_memory_carveout(self):
        """On devices that have a unified L1 cache and shared memory,
        indicates the fraction to be used for shared memory as a
        `percentage` of the total. If the fraction does not exactly equal a
        supported shared memory capacity, then the next larger supported
        capacity is used. Can be set.
        """
        attr = driver.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
        return driver.funcGetAttribute(attr, self.kernel.ptr)

    @preferred_shared_memory_carveout.setter
    def preferred_shared_memory_carveout(self, fraction):
        attr = driver.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
        driver.funcSetAttribute(self.kernel.ptr, attr, fraction)

    def compile(self, log_stream=None):
        """Compile the current kernel.

        In general, you don't have to call this method;
        kernels are compiled implicitly on the first call.

        Args:
            log_stream (object): Pass either ``sys.stdout`` or a file object to
                which the compiler output will be written.
                Defaults to ``None``.
        """
        # Flush the cache when compilation is explicitly requested
        self._kernel_cache = [None] * runtime.getDeviceCount()
        self._kernel(log_stream=log_stream)


cdef class RawModule:
    """User-defined custom module.

    This class can be used to either compile raw CUDA sources or load CUDA
    modules (\\*.cubin, \\*.ptx). This class is useful when a number of CUDA
    kernels in the same source need to be retrieved.

    For the former case, the CUDA source code is compiled when any method is
    called. For the latter case, an existing CUDA binary (\\*.cubin) or a PTX
    file can be loaded by providing its path.

    CUDA kernels in a :class:`RawModule` can be retrieved by calling
    :meth:`get_function`, which will return an instance of :class:`RawKernel`.
    (Same as in :class:`RawKernel`, the generated binary is also cached.)

    Args:
        code (str): CUDA source code. Mutually exclusive with ``path``.
        path (str): Path to cubin/ptx. Mutually exclusive with ``code``.
        options (tuple of str): Compiler options passed to the backend (NVRTC
            or NVCC). For details, see
            https://docs.nvidia.com/cuda/nvrtc/index.html#group__options or
            https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#command-option-description.
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
        name_expressions (sequence of str): A sequence (e.g. list) of strings
            referring to the names of C++ global/template kernels. For example,
            ``name_expressions=['func1<int>', 'func1<double>', 'func2']`` for
            the template kernel ``func1<T>`` and non-template kernel ``func2``.
            Strings in this tuple must then be passed, one at a time, to
            :meth:`get_function` to retrieve the corresponding kernel.
        jitify (bool): Whether or not to use `Jitify`_ to assist NVRTC to
            compile C++ kernels. Defaults to ``False``.

    .. note::
        Each kernel in ``RawModule`` possesses independent function attributes.

    .. note::
        Before CuPy v8.0.0, the compilation happens at initialization. Now, it
        happens at the first time retrieving any object (kernels, pointers, or
        texrefs) from the module.

    .. _Jitify:
        https://github.com/NVIDIA/jitify

    """
    def __init__(self, *, str code=None, str path=None, tuple options=(),
                 str backend='nvrtc', bint translate_cucomplex=False,
                 bint enable_cooperative_groups=False,
                 name_expressions=None, bint jitify=False):
        if (code is None) == (path is None):
            raise TypeError(
                'Exactly one of `code` and `path` keyword arguments must be '
                'given.')
        if name_expressions is not None:
            if code is None:
                raise ValueError('need CUDA C++ code for the requested '
                                 'kernels')
            if backend != 'nvrtc':
                raise ValueError('only nvrtc supports retrieving the mangled '
                                 'names for the given name expressions')
            for option in options:
                if '-std=c++' in option:  # both -std and --std are valid
                    break
            else:
                raise ValueError('need to specify C++ standard for compiling '
                                 'template code')
            self.name_expressions = tuple(name_expressions)  # make it hashable
        else:
            self.name_expressions = None
        if jitify:
            if code is None:
                raise ValueError('Jitify does not support precompiled objects')
            if backend != 'nvrtc':  # TODO(leofang): how about hiprtc?
                raise ValueError('Jitify only supports NVRTC')

        self.code = code
        self.file_path = path
        self.enable_cooperative_groups = enable_cooperative_groups
        self.jitify = jitify

        if self.code is not None:
            self.options = options
            self.backend = backend
            self.translate_cucomplex = translate_cucomplex
        elif self.file_path is not None:
            self.options = ()
            self.backend = 'nvcc'
            self.translate_cucomplex = False

    @property
    def module(self):
        return self._module()

    def _module(self, log_stream=None):
        # The module is cached, so on the device where this has been called,
        # we would just look up from the cache, and do recompiling only when
        # switching to a different device
        cdef Module mod

        mod = _get_raw_module(
            self.code, self.file_path, self.options, self.backend,
            self.translate_cucomplex, self.enable_cooperative_groups,
            self.jitify, self.name_expressions, log_stream)
        return mod

    def compile(self, log_stream=None):
        """Compile the current module.

        In general, you don't have to call this method;
        kernels are compiled implicitly on the first call.

        Args:
            log_stream (object): Pass either ``sys.stdout`` or a file object to
                which the compiler output will be written.
                Defaults to ``None``.

        .. note::
            Calling :meth:`compile` will reset the internal state of
            a :class:`RawKernel`.

        """
        self._module(log_stream)

    def get_function(self, str name):
        """Retrieve a CUDA kernel by its name from the module.

        Args:
            name (str): Name of the kernel function. For C++ global/template
                kernels, ``name`` refers to one of the name expressions
                specified when initializing the present :class:`RawModule`
                instance.

        Returns:
            RawKernel: An ``RawKernel`` instance.

        .. note::
            The following example shows how to retrieve one of the specialized
            C++ template kernels:

            .. code-block:: python

                code = r'''
                template<typename T>
                __global__ void func(T* in_arr) { /* do something */ }
                '''

                kers = ('func<int>', 'func<float>', 'func<double>')
                mod = cupy.RawModule(code=code, options=('--std=c++11',),
                                     name_expressions=kers)

                // retrieve func<int>
                ker_int = mod.get_function(kers[0])

        .. seealso::
            ``nvrtcAddNameExpression`` and ``nvrtcGetLoweredName`` from
            `Accessing Lowered Names`_ of the NVRTC documentation.

        .. _Accessing Lowered Names:
            https://docs.nvidia.com/cuda/nvrtc/index.html#accessing-lowered-names

        """
        cdef RawKernel ker
        cdef Function func
        cdef str mangled_name

        # check if the name is a valid C++ name expression
        if self.name_expressions:
            mangled_name = self.module.mapping.get(name)
            if mangled_name is not None:
                name = mangled_name

        ker = RawKernel(
            self.code, name, self.options, self.backend,
            translate_cucomplex=self.translate_cucomplex,
            enable_cooperative_groups=self.enable_cooperative_groups,
            jitify=self.jitify)

        # for lookup in case we loaded from cubin/ptx
        ker.file_path = self.file_path
        # for lookup in case we specialize a template
        ker.name_expressions = self.name_expressions
        # register the kernel in the cache
        func = ker.kernel  # noqa
        return ker

    def get_texref(self, name):
        '''Retrieve a texture reference by its name from the module.

        Args:
            name (str): Name of the texture reference.

        Returns:
            intptr_t: A ``CUtexref`` handle, to be passed to :class:`~cupy.cuda.texture.TextureReference`.
        '''  # noqa
        return self.module.get_texref(name)

    def get_global(self, name):
        '''Retrieve a pointer to a global symbol by its name from the module.

        Args:
            name (str): Name of the global symbol.

        Returns:
            ~cupy.cuda.MemoryPointer: A handle to the global symbol.

        .. note::
            This method can be used to access, for example, constant memory:

            .. code-block:: python

                # to get a pointer to "arr" declared in the source like this:
                # __constant__ float arr[10];
                memptr = mod.get_global("arr")
                # ...wrap it using cupy.ndarray with a known shape
                arr_ndarray = cp.ndarray((10,), cp.float32, memptr)
                # ...perform data transfer to initialize it
                arr_ndarray[...] = cp.random.random((10,), dtype=cp.float32)
                # ...and arr is ready to be accessed by RawKernels

        '''
        from cupy.cuda.memory import MemoryPointer, UnownedMemory
        cdef Module mod = self.module
        ptr = mod.get_global_var(name)
        # 1. unable to retrieve size, plus it's not used anywhere, so set to 0
        # 2. it is safe to call getDevice() since self.module is cached on a
        #    per-device basis
        # 3. in CUDA, passing the device id saves us a look-up of the pointer
        #    attributes; in ROCm, this is a must because there's a bug when
        #    looking up a pointer to constant memory (hipErrorInvalidDevice)
        cdef int dev = runtime.getDevice()
        mem = UnownedMemory(ptr, 0, mod, dev)
        memptr = MemoryPointer(mem, 0)
        return memptr


@cupy._util.memoize(for_each_device=True)
def _get_raw_module(str code, str path, tuple options, str backend,
                    bint translate_cucomplex,
                    bint enable_cooperative_groups,
                    bint jitify,
                    tuple name_expressions,
                    object log_stream):
    cdef Module mod
    if code is not None:
        mod = cupy._core.core.compile_with_cache(
            code, options, prepend_cupy_headers=False, backend=backend,
            translate_cucomplex=translate_cucomplex,
            enable_cooperative_groups=enable_cooperative_groups,
            name_expressions=name_expressions,
            log_stream=log_stream, jitify=jitify)
    elif path is not None:
        mod = Module()
        mod.load_file(path)
    return mod
