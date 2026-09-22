"""Minimal public Cython API.

Outside of bindings (for which moving to cuda-python/nvmath-python is
preferable), CuPy's Cython API is not considered public.

This `.pxd` file exposes minimal, fast access. Note that including this
`.pxd` file will cause `cupy` to be imported. If this is not desirable
it is your responsibility to create a layout where you e.g. check
for `type.__module__.split(".")[0] == "cupy"` prior to importing the module
which imports it (or use the raw C-API).

The API includes backwards compatibility helpers in the header and
works with older CuPy versions. To use this API, you must add CuPy's
`include` directory to the include path or vendor the header locally.

.. note::
    It is your responsibility to ensure compatibility in case of future
    API extensions (or changes). In case of future API additions you must
    check the embedded version before using them.

Usage
-----
To use this file::

    from cupy.public_c_api cimport CuPyAPI, get_cupy_api, ndarray


    cdef CuPyAPI *cp_api = get_cupy_api()

    def use_array(arr):
        cdef void *ptr
        cp_api.get_ndarray_ptr(arr, &ptr)
        # or, if you know it is a cupy ndarray:
        cp_api.get_ndarray_ptr(<ndarray>arr, &ptr)

Cython type-checks the first form for you.  `None` is the one exception:
it passes the check and is then cast like any other object, so declare
arguments as `ndarray not None` where you can.
"""

from cpython cimport PyObject, PyTypeObject

# The C header leaves CuPyNDArray incomplete (abi3t-safe). But Cython
# currently needs a complete type.  Insert that here.
cdef extern from *:
    """
    struct CuPyNDArray { PyObject *obj; };
    """


cdef extern from "cupy_public_c_api.h":

    ctypedef struct CuPyNDArrayMetadata:
        void *ptr
        int ndim
        int device_id
        size_t size
        Py_ssize_t itemsize
        const Py_ssize_t *shape
        const Py_ssize_t *strides  # array strides in bytes
        PyObject *dtype  # NumPy dtype.

    ctypedef class cupy.ndarray [object CuPyNDArray, check_size ignore]:
        """Opaque cupy.ndarray (C: ``CuPyNDArray``)."""
        pass

    ctypedef struct CuPyAPI:
        int version_major
        int version_minor
        PyTypeObject *ndarray_type  # the `cupy.ndarray` type.
        # Functions available on all CuPy versions.
        int get_ndarray_ptr(ndarray, void **) except -1
        int get_ndarray_metadata(ndarray, CuPyNDArrayMetadata *) except -1
        # Functions available on CuPy >= 14.3
        int get_current_stream_ptr(int device_id, void **) except -1

    cdef CuPyAPI *get_cupy_api() except NULL


# NOTE(seberg): This file should not define explicit Cython API, there
# is currently no Cython exposure (all of these are Cython helpers for C/C++
# exposure via `cupy_public_c_api.h`).
# Everything here is `extern`, so there is deliberately no `cupy.public_c_api`
# module to go with it; the capsule is built by `cupy._core.public_c_api`.
