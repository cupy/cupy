:orphan:

.. _C_memory_API:

Memory management in C/C++
--------------------------

For building Python modules in C/C++, CuPy provides C API to access its current
device memory allocator. To use this, please include ``cupy/cuda/cupy_memory.h``
in your C/C++ code.

.. c:function:: cupy_allocator_handle* get_cupy_allocator_handle()

    Creates an opaque handle to CuPy's *current* device memory allocator.

    .. note::

        Internally, it inspects if the Python interpreter is already initialized
        in the current thread, calls ``Py_Initialize()`` if not, and import the
        necessary functions from :mod:`cupy.cuda.memory` for use.

.. c:function:: void destroy_cupy_allocator_handle(cupy_allocator_handle* ptr)

    Destroys the handle ``ptr`` to CuPy's device memory allocator.

    Any device memory allocated with the handle should be freed before destroying
    it.

    .. note::

        Internally, it checks whether ``Py_Initialize()`` was called when the
        handle was created, and calls ``Py_Finalize()`` if so.

.. c:function:: void* cupy_malloc(cupy_allocator_handle* handle, size_t size)

    Allocates device memory of ``size`` bytes from CuPy's memory pool.

    The device on which the memory is allocated depends on the current CUDA
    context, so callers should ensure ``cudaSetDevice()`` is called prior to
    allocating memory.

    .. note::

        This function can only be called on the host, as the Python GIL is
        hold during the call.

.. c:function:: void cupy_free(cupy_allocator_handle* handle, void* ptr)

    Frees the allocated CuPy memory pointed by ``ptr``.

    .. note::

        This function can only be called on the host, as the Python GIL is
        hold during the call.
