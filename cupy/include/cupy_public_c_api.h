#ifndef CUPY_PUBLIC_C_API_H
#define CUPY_PUBLIC_C_API_H

/*
 * Minimal public C API for CuPy ndarrays.
 *
 * Consumers obtain a `CuPyAPI` table via `get_cupy_api()`.  The table is
 * exported at runtime as the `cupy._public_c_api` capsule (capsule name
 * `"cupy._public_c_api"`).  `get_cupy_api()` is a thin wrapper around
 * `PyCapsule_Import`; unlike NumPy's `import_array()` it performs no
 * version checks of its own.  Callers must inspect `version_major` /
 * `version_minor` before using functions added in later releases.
 *
 * `CuPyAPI` is append-only (new function pointers at the end) although
 * existing functions could transition to always error and, if necessary,
 * a transition could happen via deprecation or error in `__getattr__`).
 *
 * On C++ (except when targeting abi3t), if the capsule is missing and
 * `cupy.__version__` is in `[8.0, 14.3)`, a fallback table is used.
 * (That path relies on the CuPy ABI being stable in that range.)
 * Not every slot is filled there: `get_current_stream_ptr` is NULL
 * until CuPy 14.3.
 *
 * This API assumes the GIL is held and functions return -1 with a
 * Python error set on failure.
 */

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CuPyNDArrayMetadata {
    void *ptr;
    int ndim;
    int device_id;
    size_t size;
    Py_ssize_t itemsize;
    const Py_ssize_t *shape;
    const Py_ssize_t *strides;
    PyObject *dtype;  // borrowed reference to numpy.dtype (PyArray_Descr)
} CuPyNDArrayMetadata;

/*
 * Opaque ndarray.  Cast from PyObject *; the cast itself is unchecked, use
 * `PyObject_TypeCheck(obj, api->ndarray_type)` first unless the type is known.
 */
typedef struct CuPyNDArray CuPyNDArray;

/*
 * The CuPy public API table.  All functions return -1 with a Python error
 * set on failure and the caller must check (even if the functions may never
 * fail in practice).
 */
typedef struct CuPyAPI {
    int version_major;
    int version_minor;
    PyTypeObject *ndarray_type;
    /* Functions available on all CuPy versions. */
    int (*get_ndarray_ptr)(CuPyNDArray *arr, void **ptr);
    int (*get_ndarray_metadata)(CuPyNDArray *arr, CuPyNDArrayMetadata *out);
    /* Functions available on CuPy >=14.3 */
    int (*get_current_stream_ptr)(int device_id, void **ptr);
} CuPyAPI;

static inline CuPyAPI *get_cupy_api(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif


#if defined(__cplusplus) && !defined(Py_TARGET_ABI3T)
/*
 * Legacy fallback for CuPy <14.3.
 *
 * The below functionality provides direct C API access to CuPy ndarrays
 * for CuPy versions < 14.3.
 * This works in practice because we know that CuPy 8-14.3 were ABI stable.
 *
 * The fallback path needs `std::vector` and `PyObject` internals, so is only
 * defined in C++ and not compiling for the stable ABI (limited API is fine).
 *
 * When we can assume almost all downstream uses CuPy 14.3+ this fallback
 * should be removed.
 */

#include <cstdio>
#include <vector>

struct _CuPyMemoryPointerObject {
    PyObject_HEAD
    void *__pyx_vtab;
    intptr_t ptr;
    int device_id;
    PyObject *mem;
};

struct _CuPyNDArrayObject {
    PyObject_HEAD
    void *__pyx_vtab;
    PyObject *__weakref__;
    Py_ssize_t size;
    std::vector<Py_ssize_t> _shape;
    std::vector<Py_ssize_t> _strides;
    int _c_contiguous;
    int _f_contiguous;
    int _index_32_bits;
    PyObject *dtype;
    _CuPyMemoryPointerObject *data;
    PyObject *base;
};


extern "C" {

static inline int
_cupy_legacy_get_ndarray_ptr(CuPyNDArray *arr, void **ptr)
{
    _CuPyNDArrayObject *obj = (_CuPyNDArrayObject *)arr;
    *ptr = (void *)obj->data->ptr;
    return 0;
}

static inline int
_cupy_legacy_get_ndarray_metadata(CuPyNDArray *arr, CuPyNDArrayMetadata *out)
{
    _CuPyNDArrayObject *obj = (_CuPyNDArrayObject *)arr;
    out->ptr = (void *)obj->data->ptr;
    out->ndim = (int)obj->_shape.size();
    out->device_id = obj->data->device_id;
    out->size = (size_t)obj->size;
    out->shape = obj->_shape.data();
    out->strides = obj->_strides.data();
    out->dtype = obj->dtype;
    PyObject *itemsize_obj = PyObject_GetAttrString(obj->dtype, "itemsize");
    if (itemsize_obj == NULL) {
        return -1;
    }
    out->itemsize = PyLong_AsSsize_t(itemsize_obj);
    Py_DECREF(itemsize_obj);
    if (out->itemsize == -1 && PyErr_Occurred()) {
        return -1;
    }
    return 0;
}

static inline CuPyAPI *
_get_cupy_api_legacy(void)
{
    /* Layout-based table used instead of the capsule when CuPy < 14.3.
     * get_current_stream_ptr is capsule-only (>= 14.3). */
    static CuPyAPI _cupy_api_stable = {
        0,
        0,
        NULL,
        _cupy_legacy_get_ndarray_ptr,
        _cupy_legacy_get_ndarray_metadata,
        NULL,
    };

    PyObject *cupy = PyImport_ImportModule("cupy");
    if (cupy == NULL) {
        return NULL;
    }
    PyObject *ver_obj = PyObject_GetAttrString(cupy, "__version__");
    if (ver_obj == NULL) {
        Py_DECREF(cupy);
        return NULL;
    }
    /* PyUnicode_AsUTF8() is not part of the limited API. */
    PyObject *ver_bytes = PyUnicode_AsUTF8String(ver_obj);
    Py_DECREF(ver_obj);
    if (ver_bytes == NULL) {
        Py_DECREF(cupy);
        return NULL;
    }
    const char *ver = PyBytes_AsString(ver_bytes);
    if (ver == NULL) {
        Py_DECREF(ver_bytes);
        Py_DECREF(cupy);
        return NULL;
    }
    int major = 0;
    int minor = 0;
    if (std::sscanf(ver, "%d.%d", &major, &minor) < 2) {
        PyErr_Format(PyExc_RuntimeError,
                     "unrecognized cupy.__version__: %s", ver);
        Py_DECREF(ver_bytes);
        Py_DECREF(cupy);
        return NULL;
    }
    Py_DECREF(ver_bytes);

    if (major < 8) {
        /* Not plausible to reach: CuPy 8 was released in 2020, Python 3.10
         * (the oldest one can build for here) in 2021. */
        PyErr_Format(
            PyExc_RuntimeError,
            "cupy %d.%d is too old for direct C API access (need >= 8.0)",
            major, minor);
        Py_DECREF(cupy);
        return NULL;
    }
    if (major > 14 || (major == 14 && minor >= 3)) {
        /* Capsule is required; get_cupy_api restores the import error. */
        Py_DECREF(cupy);
        return NULL;
    }

    PyObject *ndarray_type = PyObject_GetAttrString(cupy, "ndarray");
    Py_DECREF(cupy);
    if (ndarray_type == NULL) {
        return NULL;
    }
    if (!PyType_Check(ndarray_type)) {
        Py_DECREF(ndarray_type);
        PyErr_SetString(PyExc_RuntimeError, "cupy.ndarray is not a type");
        return NULL;
    }
    if (_cupy_api_stable.ndarray_type == NULL) {
        /* Reference kept on purpose: the table lives until process exit. */
        _cupy_api_stable.ndarray_type = (PyTypeObject *)ndarray_type;
    }
    else {
        Py_DECREF(ndarray_type);
    }
    _cupy_api_stable.version_major = major;
    _cupy_api_stable.version_minor = minor;
    return &_cupy_api_stable;
}

}  /* extern "C" */

#endif  /* __cplusplus && !Py_TARGET_ABI3T */


#ifdef __cplusplus
extern "C" {
#endif

/*
 * Main public API entry-point.  The main use-case for this is to enable the
 * legacy fallback. If that fallback is not required, `PyCapsule_Import` does
 * the same thing.
 */
static inline CuPyAPI *
get_cupy_api(void)
{
    CuPyAPI *api = (CuPyAPI *)PyCapsule_Import("cupy._public_c_api", 0);
    if (api != NULL) {
        return api;
    }
#if defined(__cplusplus) && !defined(Py_TARGET_ABI3T)
    if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyObject *exc_type, *exc_value, *exc_tb;
        PyErr_Fetch(&exc_type, &exc_value, &exc_tb);
        api = _get_cupy_api_legacy();
        if (api == NULL && !PyErr_Occurred()) {
            /* Version >= 14.3: keep the original capsule-import error. */
            PyErr_Restore(exc_type, exc_value, exc_tb);
            return NULL;
        }
        Py_XDECREF(exc_type);
        Py_XDECREF(exc_value);
        Py_XDECREF(exc_tb);
        return api;
    }
#endif
    return NULL;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* CUPY_PUBLIC_C_API_H */
