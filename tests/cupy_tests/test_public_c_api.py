"""
Tests for the public C API capsule via the Cython helper
definitions.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import sysconfig

import numpy
import pytest

import cupy
from cupy import testing


_INCLUDE_DIR = os.path.join(os.path.dirname(cupy.__file__), 'include')

_PUBLIC_API_PYX = r'''
# distutils: language = c++

from libc.stdint cimport intptr_t

from cupy.public_c_api cimport (
    CuPyAPI, CuPyNDArrayMetadata, get_cupy_api, ndarray)

cdef extern from "cupy_public_c_api.h":
    # Private C++ fallback used when CuPy < 14.3 has no capsule.  Tested
    # here so that a layout change is caught while it is still supported.
    int _cupy_legacy_get_ndarray_ptr(ndarray, void **) except -1
    int _cupy_legacy_get_ndarray_metadata(
        ndarray, CuPyNDArrayMetadata *) except -1


cdef CuPyAPI *cp_api = get_cupy_api()


def api_version():
    return cp_api.version_major, cp_api.version_minor


def get_ndarray_type():
    return <object>cp_api.ndarray_type


def get_ptr(arr):
    cdef void *ptr
    cp_api.get_ndarray_ptr(arr, &ptr)
    return <intptr_t>ptr


cdef object _as_tuple(CuPyNDArrayMetadata *meta):
    cdef Py_ssize_t i
    return (
        <intptr_t>meta.ptr,
        meta.ndim,
        meta.device_id,
        meta.size,
        tuple([meta.shape[i] for i in range(meta.ndim)]),
        tuple([meta.strides[i] for i in range(meta.ndim)]),
        <object>meta.dtype,
        meta.itemsize,
    )


def get_metadata(arr):
    cdef CuPyNDArrayMetadata meta
    cp_api.get_ndarray_metadata(arr, &meta)
    return _as_tuple(&meta)


def get_ptr_legacy(arr):
    cdef void *ptr
    _cupy_legacy_get_ndarray_ptr(arr, &ptr)
    return <intptr_t>ptr


def get_metadata_legacy(arr):
    cdef CuPyNDArrayMetadata meta
    _cupy_legacy_get_ndarray_metadata(arr, &meta)
    return _as_tuple(&meta)


def get_current_stream_ptr(device_id):
    cdef void *ptr
    cp_api.get_current_stream_ptr(device_id, &ptr)
    return <intptr_t>ptr
'''


def _run_build_ext(tmp_path, ext_modules, name):
    from setuptools import setup

    cwd = os.getcwd()
    old_argv = sys.argv
    try:
        os.chdir(tmp_path)
        sys.argv = ['setup.py', 'build_ext', '--inplace']
        setup(name=name, ext_modules=ext_modules)
    finally:
        sys.argv = old_argv
        os.chdir(cwd)


def _compile_cython_ext(
        tmp_path, name, source, include_dirs=(), limited_api=False):
    from Cython.Build import cythonize
    from setuptools import Extension

    define_macros = []
    if limited_api:
        define_macros = [
            ('CYTHON_LIMITED_API', '1'),
            ('Py_LIMITED_API', '0x030a0000'),
        ]

    tmp_path.mkdir(parents=True, exist_ok=True)
    pyx = tmp_path / f'{name}.pyx'
    pyx.write_text(source)
    _run_build_ext(
        tmp_path,
        cythonize(
            [Extension(
                name,
                sources=[str(pyx)],
                include_dirs=list(include_dirs),
                define_macros=define_macros,
                py_limited_api=limited_api,
                language='c++',
            )],
            language_level='3',
            quiet=True,
            nthreads=0,
        ),
        name=name,
    )
    # `.abi3.so` for a limited API build, `EXT_SUFFIX` otherwise.
    built, = (p for p in tmp_path.glob(f'{name}*')
              if p.suffix not in ('.pyx', '.cpp'))
    return built


def _compile_cython_public_api_mod(tmp_path, limited_api):
    ext_path = _compile_cython_ext(
        tmp_path,
        'cupy_public_api_testmod',
        _PUBLIC_API_PYX,
        include_dirs=[_INCLUDE_DIR],
        limited_api=limited_api,
    )
    spec = importlib.util.spec_from_file_location(
        'cupy_public_api_testmod', ext_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def cython_api(tmp_path_factory):
    pytest.importorskip('Cython')
    pytest.importorskip('setuptools')
    # Build against the limited API where we can, it is the stricter of
    # the two.  Not on free-threaded builds: that is abi3t, which the
    # header opts out of (and <3.15 rejects the combination anyway).
    limited_api = not sysconfig.get_config_var('Py_GIL_DISABLED')
    tmp_path = tmp_path_factory.mktemp('public_api')
    try:
        _compile_cython_ext(
            tmp_path / 'probe',
            'cupy_cxx_probe',
            '# distutils: language = c++\n',
            limited_api=limited_api,
        )
    except Exception as exc:
        pytest.skip(f'Build toolchain not usable: {exc}')
    return _compile_cython_public_api_mod(tmp_path / 'mod', limited_api)


class TestPublicCAPI:

    def test_header_is_installed(self):
        header = os.path.join(_INCLUDE_DIR, 'cupy_public_c_api.h')
        assert os.path.isfile(header)

    def test_capsule_is_exported(self):
        assert hasattr(cupy, '_public_c_api')

    def test_version(self, cython_api):
        major, minor = (int(p) for p in cupy.__version__.split('.')[:2])
        assert cython_api.api_version() == (major, minor)

    @testing.for_all_dtypes()
    def test_ndarray_ptr_and_metadata(self, cython_api, dtype):
        arr = cupy.zeros((2, 3), dtype=dtype)
        assert cython_api.get_ptr(arr) == arr.data.ptr

        ptr, ndim, device_id, size, shape, strides, dt, itemsize = (
            cython_api.get_metadata(arr))
        assert ptr == arr.data.ptr
        assert ndim == arr.ndim
        assert device_id == arr.data.device_id
        assert size == arr.size
        assert shape == arr.shape
        assert strides == arr.strides
        assert dt is arr.dtype
        assert itemsize == arr.dtype.itemsize

    def test_shape_and_strides_3d(self, cython_api):
        arr = cupy.empty((2, 3, 4), dtype=cupy.float32)
        _, _, _, _, shape, strides, _, _ = cython_api.get_metadata(arr)
        assert shape == arr.shape
        assert strides == arr.strides

    def test_view(self, cython_api):
        # A view has an offset pointer and non-contiguous strides.
        view = cupy.empty((4, 6), dtype=cupy.float64)[1:, ::2]
        ptr, _, _, size, shape, strides, _, _ = (
            cython_api.get_metadata(view))
        assert ptr == view.data.ptr
        assert size == view.size
        assert shape == view.shape
        assert strides == view.strides

    def test_legacy_fallback_matches(self, cython_api):
        # The C++ fallback pokes at the ndarray struct directly, it must
        # agree with the capsule for as long as it is supported.
        arr = cupy.empty((4, 6), dtype=cupy.float64)[1:, ::2]
        assert cython_api.get_ptr_legacy(arr) == arr.data.ptr
        assert (cython_api.get_metadata_legacy(arr)
                == cython_api.get_metadata(arr))

    def test_0d(self, cython_api):
        arr = cupy.array(1.5)
        _, ndim, _, size, shape, strides, _, _ = cython_api.get_metadata(arr)
        assert ndim == 0
        assert size == 1
        assert shape == ()
        assert strides == ()

    def test_typecheck(self, cython_api):
        # The Cython API will type-check (unless manually cast to `ndarray`)
        with pytest.raises(TypeError):
            cython_api.get_ptr(numpy.zeros((2, 3)))

    def test_ndarray(self, cython_api):
        assert cython_api.get_ndarray_type() is cupy.ndarray

    def test_current_stream_ptr(self, cython_api):
        device_id = cupy.cuda.Device().id
        # Set temporary stream and check that get_current_stream_ptr matches:
        with cupy.cuda.Stream() as stream:
            assert cython_api.get_current_stream_ptr(device_id) == stream.ptr

        with pytest.raises(ValueError):
            cython_api.get_current_stream_ptr(-1)
