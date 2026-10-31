.. _public_c_api:

Public C API
============

CuPy exposes a small C API for reading :class:`~cupy.ndarray` metadata and the
current CUDA stream from C, C++, or Cython without going through Python
attribute lookups.

This is *not* a general Cython API.  Other ``.pxd`` files under ``cupy``
are private.

Note that while we do not anticipate changes especially for already build packages,
this API is considered unstable.

Getting the header and API table
--------------------------------

The header lives in CuPy's ``include`` directory::

    include_dir = os.path.join(os.path.dirname(cupy.__file__), 'include')

Add it to the compiler include path, then you can include the header::

    #include "cupy_public_c_api.h"

For Cython, the following convenience wrapper is defined (mirroring the C-API)::

    from cupy.public_c_api cimport CuPyAPI, get_cupy_api, ndarray

    cdef CuPyAPI *api = get_cupy_api()

Where ``get_cupy_api()`` imports the ``cupy._public_c_api`` capsule (no version
check, unlike NumPy's ``import_array()``).  The header also has a C++
fallback for older CuPy (not abi3t).

The header and ``.pxd`` file can be vendored into downstream projects and will
work with any CuPy version (see below).

.. versionadded::
    Added in CuPy 14.3. So that CuPy >=14.3 is a build (not runtime)
    requirement for obtaining the header.

API functionality
-----------------

.. warning::
    Check ``version_major`` / ``version_minor`` before using later slots.

``CuPyAPI`` is a struct which contains the following entries. Later fields
may only be defined in new versions of CuPy and all access must be guarded
by a version check:

- ``ndarray_type``, the ``cupy.ndarray`` type (all versions).  Borrowed and
  valid for as long as CuPy is imported.  Use it with
  ``PyObject_TypeCheck()`` to also accept subclasses.
- ``get_ndarray_ptr(CuPyNDArray *arr, void **ptr)`` (all versions)
- ``get_ndarray_metadata(CuPyNDArray *arr, CuPyNDArrayMetadata *out)``
  (all versions; see the header for the metadata struct)
- ``get_current_stream_ptr(int device_id, void **ptr)`` (CuPy >= 14.3).
  The stream is only valid in the current context.

``ndarray`` / ``CuPyNDArray *`` are opaque and the cast from ``PyObject *``
is unchecked, so C/C++ callers should use
``PyObject_TypeCheck(obj, api->ndarray_type)`` when the type is not already
known.  Cython inserts that check for you, except for ``None`` (declare
arguments ``ndarray not None``) or when you cast with ``<ndarray>``.

The GIL must be held.  Functions return ``-1`` with a Python error on
failure.  Keep the array alive while using borrowed data.
