# distutils: language = c++
"""Implementation of the public C API capsule ``cupy._public_c_api``.

The declarations live in ``cupy/public_c_api.pxd``, which is the public
(cimport-able) name.  That file is declaration-only, so it does not need
a module of its own and this one can stay private.
"""

from cpython.pycapsule cimport PyCapsule_New

cimport numpy as cnp

from cupy.public_c_api cimport *
from cupy._core.core cimport _ndarray_base
from cupy_backends.cuda cimport stream as _backends_stream
from cupy._version import __version__ as _cupy_version


cdef int _get_ndarray_ptr(ndarray arr, void **ptr) except -1:
    cdef _ndarray_base a = <_ndarray_base>arr
    ptr[0] = <void *>a.data.ptr
    return 0


cdef int _get_ndarray_metadata(
        ndarray arr, CuPyNDArrayMetadata *out) except -1:
    cdef _ndarray_base a = <_ndarray_base>arr
    out.ptr = <void *>a.data.ptr
    out.ndim = <int>a._shape.size()
    out.device_id = a.data.device_id
    out.size = <size_t>a.size
    out.itemsize = (<cnp.dtype>a.dtype).itemsize
    out.shape = a._shape.data()
    out.strides = a._strides.data()
    out.dtype = <PyObject *>a.dtype
    return 0


cdef int _get_current_stream_ptr(int device_id, void **ptr) except -1:
    # -1 (i.e. "the current device") is not accepted: the caller has to call
    # cudaGetDevice() anyway, so accepting it just risks calling it twice.
    if device_id < 0:
        raise ValueError("get_current_stream_ptr: device_id must be >= 0")
    ptr[0] = <void *>_backends_stream.get_stream_ptr(device_id)
    return 0


cdef CuPyAPI _cupy_api
_cupy_api.version_major = int(_cupy_version.split('.')[0])
_cupy_api.version_minor = int(_cupy_version.split('.')[1])
_cupy_api.ndarray_type = <PyTypeObject *>ndarray
_cupy_api.get_ndarray_ptr = _get_ndarray_ptr
_cupy_api.get_ndarray_metadata = _get_ndarray_metadata
_cupy_api.get_current_stream_ptr = _get_current_stream_ptr

_public_c_api = PyCapsule_New(&_cupy_api, "cupy._public_c_api", NULL)
