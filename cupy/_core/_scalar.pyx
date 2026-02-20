from libc.stdint cimport intptr_t

cimport numpy as cnp

import graphlib
import hashlib
import textwrap

import numpy

import cupy
from cupy._core cimport _dtype


cdef extern from 'numpy/ndarraytypes.h':
    cdef int PyArray_Pack(cnp.dtype dtype, void *ptr, object value) except -1


cdef dict _typenames_base = {
    numpy.dtype('float64'): ('double', None),
    numpy.dtype('float32'): ('float', None),
    numpy.dtype('float16'): ('float16', '#include "cupy/float16.cuh"'),
    numpy.dtype('complex128'): ('thrust::complex<double>', None),
    numpy.dtype('complex64'): ('thrust::complex<float>', None),
    numpy.dtype('int64'): ('long long', None),
    numpy.dtype('int32'): ('int', None),
    numpy.dtype('int16'): ('short', None),
    numpy.dtype('int8'): ('signed char', None),
    numpy.dtype('uint64'): ('unsigned long long', None),
    numpy.dtype('uint32'): ('unsigned int', None),
    numpy.dtype('uint16'): ('unsigned short', None),
    numpy.dtype('uint8'): ('unsigned char', None),
    numpy.dtype('bool'): ('bool', None),
}


cdef object _numpy_bool = numpy.dtype(numpy.bool_)
cdef object _numpy_int32 = numpy.dtype(numpy.int32)
cdef object _numpy_int64 = numpy.dtype(numpy.int64)
cdef object _numpy_uint64 = numpy.dtype(numpy.uint64)
cdef object _numpy_float64 = numpy.dtype(numpy.float64)
cdef object _numpy_complex128 = numpy.dtype(numpy.complex128)


cdef _flatten_type_decls(type_decls, dict declarations):
    cdef decl
    cdef frozenset decl_deps
    cdef set direct_deps = set()

    # NOTE(seberg): It would be strange if two entries weren't identical
    # in what they depend on. So we don't guard against it here.
    for decl in type_decls:
        if isinstance(decl, str):
            direct_deps.add(decl)
            declarations[decl] = ()
        elif isinstance(decl, tuple):
            decl, decl_deps = decl

            direct_deps.add(decl)
            declarations[decl] = _flatten_type_decls(decl_deps, declarations)
        else:
            raise TypeError("type_decls must be str or (str, frozenset)")

    return direct_deps


cpdef str format_type_decls(set type_decls):
    """
    When using `type_decls` to support e.g. header specific types and
    structured dtype declarations, we would like the result to be stable
    so e.g. caching cannot be disturbed.
    This function does the right formatting/flattening and pairs with
    `get_typename`.  It returns either an empty string or a correct code
    block (with two trailing newlines to separate it from what follows).
    """
    if not type_decls:
        return ""

    # Formatting the is unfortunately not as simple as `sorted()` as it can
    # be nested, etc.
    cdef dict declarations = {}
    # Recursively flatten the type declarations into a dict
    _flatten_type_decls(type_decls, declarations)
    # Sort the dictionary by it's keys (to achieve a stable order)
    declarations = dict(sorted(declarations.items()))

    ts = graphlib.TopologicalSorter(declarations)
    return "\n".join(ts.static_order()) + '\n\n'


cpdef str get_typename(dtype, type_decls=None):
    """Fetch the C type name. Note that some names may require
    additionally headers to be included in order to be available.

    If not None, `type_decls` must be a set and the dtype preamble
    (i.e. this should be required headers) will be inserted.
    A preamble is either a string or a tuple of (string, frozenset)
    where the frozenset is also a set of `type_decls` (the the first
    depends on).

    If you just have a header, order should normally not matter so you
    can just pass a string. It matters for structured dtypes that
    need their field declaration to come first.
    """
    cdef cnp.dtype descr
    if dtype is None:
        raise TypeError('dtype is None')

    # TODO: Fix and make fast, using NumPy C-API.
    info = _typenames.get(dtype, None)
    if info is None:
        sctype = _dtype.get_dtype(dtype).type
        info = _typenames.get(sctype, None)

    if info is not None:
        name, header = info
        if type_decls is not None and header is not None:
            type_decls.add(header)
        return name
    elif isinstance(dtype, numpy.dtype):
        descr = <cnp.dtype>dtype
        if descr.type_num == cnp.NPY_VOID:
            if cnp.PyDataType_HASFIELDS(descr):
                # NOTE: Caching this may not be trivial if we use metadata.
                name, *_ = _build_struct_typename(dtype, type_decls)
                return name
            elif not cnp.PyDataType_HASSUBARRAY(descr):
                # Unstructured void is just a blob bytes.
                if type_decls is not None:
                    type_decls.add('#include "cupy/unstructued_void.cuh"')
                return f"cupy::UnstructuredVoid<{descr.itemsize}>"

    raise TypeError(f"Unable to find C++ type for dtype {dtype}")


cdef dict _cuda_alignments = {}
cdef object _alignment_kernel = None


cdef Py_ssize_t get_cuda_alignment(dtype) except -1:
    """Get the alignment of a given dtype within the kernel. This currently
    uses an ElementwiseKernel to compile and get the actual alignment.
    (Although, normally that should just be the itemsize.)
    """
    global _cuda_alignments, _alignment_kernel

    alignment = _cuda_alignments.get(dtype)
    if alignment is not None:
        return alignment

    if dtype.num == cnp.NPY_VOID:
        # It is unclear that this would be useful. Effectively, we would find
        # out the right alignment already as part of `get_typename` before we
        # launch the kernel.
        raise NotImplementedError(
            f"get_cuda_alignment() only supports structured dtypes, "
            f"got {dtype}")

    if _alignment_kernel is None:
        _alignment_kernel = cupy.ElementwiseKernel(
            "T in", "int64 out",
            "using in_t = decltype(in); out = alignof(in_t);",
        )

    alignment = int(_alignment_kernel(cupy.empty((), dtype=dtype))[()])
    _cuda_alignments[dtype] = alignment
    return alignment


def _build_struct_typename(dtype, type_decls):
    """Builds the struct typename and additionally returns the alignment
    (we consider the maximum alignment of any of the fields here).
    """
    # The alignment must be too small pretty much, but use it anyway.
    # If `make_aligned_dtype` was used may use __cuda_alignment__.
    alignment = dtype.alignment
    if dtype.metadata:
        # If manually overridden, use that alignment:
        alignment = dtype.metadata.get("__cuda_alignment__", alignment)
    curr_start = 0
    offsets = []
    struct_fields = []
    fields = []
    struct_compatible = True

    # subdtype_decls are dependencies for this type, we iclude the header
    # itself here as well:
    cdef set subtype_decls = set()
    subtype_decls.add('#include "cupy/structview.cuh"')

    for name, (subdtype, offset, *_) in dtype.fields.items():
        # The fields tupe can contain a 4th title, we ignore it.
        # (A title is an alternative field name...)

        # TODO(seberg): We should be able to query the JIT for the actual
        # alignment constraints (making this a trivial recursion)
        if subdtype.num == cnp.NPY_VOID and subdtype.fields is not None:
            subname, struct_name, subalignment = _build_struct_typename(
                subdtype, subtype_decls)
        else:
            subalignment = get_cuda_alignment(subdtype)
            subname = struct_name = get_typename(subdtype, subtype_decls)
            if subdtype.metadata:
                # If manually overridden, use that alignment:
                subalignment = subdtype.metadata.get(
                    "__cuda_alignment__", subalignment)

        assert (subalignment - 1) & subalignment == 0
        alignment = max(alignment, subalignment)

        if offset % subalignment != 0:
            # NOTE: We don't error earlier (for field access yet).
            raise ValueError(f"Field {name} with offset {offset} is not "
                             f"aligned to {alignment} in dtype {dtype}. "
                             f"This is not supported by CuPy please ensure "
                             "the structure is aligned. You can do so with "
                             "the `make_aligned_dtype()` helper.")

        if not (0 <= (offset - curr_start) < subalignment):
            # Addign `alignas({subalignment})` would not align with the
            # actual offset. So we cannot describe it by a struct.
            # (If the offset is larger, we could achieve this via padding)
            struct_compatible = False
        else:
            curr_start = offset

        curr_start += subdtype.itemsize

        offsets.append(offset)
        # fields are only used if all fields are indeed compatible
        struct_fields.append(
            f"  alignas({subalignment}) {struct_name} {name};")
        fields.append(f"cupy::Field<{subname}, {offset}>")

    if not struct_compatible:
        # If the struct doesn't work out, just use a single _data field.
        struct_fields = [
            f"  char _data[{dtype.itemsize}];"]

    if dtype.itemsize % alignment != 0:
        raise ValueError(
            f"Itemsize {dtype.itemsize} is not a multiple of alignment "
            f"{alignment} for dtype {dtype} and kernel launches are not "
            "supported. You can ensure compatibility with the "
            "`make_aligned_dtype()` helper (or try `align=True`).")

    struct_fields = "\n".join(struct_fields)
    hash_ = hashlib.sha1(
        struct_fields.encode("utf8"), usedforsecurity=False).hexdigest()
    struct_name = "struct_" + hash_

    # NOTE: We add this to the "headers", but the headers are always sorted
    # before use and actual includes start with `#` and go first. We must not
    # start with a space or newline, though!
    # We have to do this here, because it is a template parameter.
    # TODO(seberg): Maybe type-map should be passed through actually?!
    definition = textwrap.dedent(f"""
        struct alignas({alignment}) {struct_name} {{
        {struct_fields}
        }};
    """).lstrip("\n")  # for sorting, don't start with \n
    if type_decls is not None:
        type_decls.add((definition, frozenset(subtype_decls)))

    fields = ', '.join(fields)
    name = f"cupy::StructView<{struct_name}, {dtype.itemsize}, {fields}>"
    return name, struct_name, alignment


cdef dict _typenames = {}
cdef dict _dtype_kind_size_dict = {}


cdef _setup_type_dict():
    cdef char k
    for i in _dtype.all_type_chars:
        d = numpy.dtype(i)
        t = d.type
        _typenames[t] = _typenames_base[d]
        k = ord(d.kind)
        _dtype_kind_size_dict[t] = (k, d.itemsize)
    # CUDA types
    for t in ('cudaTextureObject_t',):
        _typenames[t] = (t, None)

    # See also _util.pyx. older NumPy versions will cause crashes if we add
    # bfloat16 loops, so don't enable it.
    if numpy.lib.NumpyVersion(numpy.__version__) >= "2.1.2":
        try:
            import ml_dtypes
        except ImportError:
            pass
        else:
            dt = numpy.dtype(ml_dtypes.bfloat16)
            _dtype_kind_size_dict[dt] = ("V", 2)
            _typenames[dt] = (
                "bfloat16", '#include "cupy/bfloat16.cuh"')
            _dtype_kind_size_dict[dt.type] = ("V", 2)
            _typenames[dt.type] = (
                "bfloat16", '#include "cupy/bfloat16.cuh"')

_setup_type_dict()


cdef set _python_scalar_type_set = {int, float, bool, complex}
cdef set _numpy_scalar_type_set = set(_typenames.keys())
cdef set scalar_type_set = _python_scalar_type_set | _numpy_scalar_type_set


# Since NumPy 2 always true unless on 32bit (before windows was outlier)
assert numpy.dtype(int) == numpy.int64

cpdef tuple numpy_dtype_from_pyscalar(x):
    # Note that isinstance(x, int) matches with bool.
    typ = type(x)
    if typ is bool:
        return _numpy_bool, False
    elif typ is float:
        return _numpy_float64, float
    elif typ is complex:
        return _numpy_complex128, complex
    elif typ is int:
        if 0x8000000000000000 <= x:
            return _numpy_uint64, int
        else:
            return _numpy_int64, int

    return None, False


cdef class CScalar(CPointer):
    """Wrapper around NumPy/Python scalars to simplify internal
    processing and make a pointer to the data cleanly available.
    This is used as arguments for kernel launches were and may
    be cast to the kernel dtype via `apply_dtype()` (currently
    this will store the value a second time when needed).
    """
    ndim = 0

    def __init__(self, value, dtype=None):
        self.value = value
        if dtype is not None:
            self.descr = numpy.dtype(dtype)
            self.weak_t = False
        else:
            self.descr, self.weak_t = numpy_dtype_from_pyscalar(value)

            if self.descr is not None:
                pass  # Python scalar was processed
            elif isinstance(value, cnp.generic):
                self.descr = value.dtype
            else:
                # Future dtypes may have scalars where this is not the case
                # but for now, it should be fine.
                raise TypeError(f'Unsupported type {type(value)}')

        self._store_c_value()

    @staticmethod
    cdef CScalar from_int32(int32_t value):
        cdef CScalar self = CScalar.__new__(CScalar)
        self.value = None
        self.descr = _numpy_int32
        self.ptr = <intptr_t><void *>(self._data)
        (<int32_t *>(self.ptr))[0] = value
        return self

    cdef _store_c_value(self):
        # If we ever support dtypes larger than this (e.g. strings)
        # we will have to introduce a conditional allocation here and
        # should memset memory to NULL (must if dtype NEEDS_INIT).
        assert self.descr.itemsize < sizeof(self._data)
        # make sure ptr points to _data.
        self.ptr = <intptr_t><void *>(self._data)

        # NOTE(seberg): This uses assignment logic, which is very subtly
        # different from casting by rejecting nan -> int. This is *only*
        # relevant for `casting="unsafe"` passed to ufuncs with `dtype=`.
        # It also means we fail for out of bound integers (NEP 50 change).
        PyArray_Pack(self.descr, <void*>(self.ptr), self.value)

    cpdef apply_dtype(self, dtype):
        cdef cnp.dtype descr = cnp.dtype(dtype)
        if descr.flags & (0x01 | 0x04):
            # Can't support this, so make sure we raise appropriate error.
            _dtype.check_supported_dtype(descr, True)
            raise RuntimeError(f"Unsupported dtype {dtype} (but not raised?)")
        if descr == self.descr:
            self.descr = descr  # update dtype, may not be identical.
            return
        if self.value is None:
            # Internal/theoretical but e.g. from_int32 has no value
            raise RuntimeError("Cannot modify dtype if value is None.")

        self.descr = descr  # modify dtype if allocation succeeded
        self._store_c_value()

    cpdef get_numpy_type(self):
        return <object>(self.descr.typeobj)  # typeobj is the C-level .type


cpdef str _get_cuda_scalar_repr(obj, dtype):
    if dtype.kind == 'b':
        return str(bool(obj)).lower()
    elif dtype.kind == 'i':
        if dtype.itemsize < 8:
            return str(int(obj))
        else:
            return str(int(obj)) + 'll'
    elif dtype.kind == 'u':
        if dtype.itemsize < 8:
            return str(int(obj)) + 'u'
        else:
            return str(int(obj)) + 'ull'
    elif dtype.kind == 'f':
        if dtype.itemsize < 8:
            if numpy.isnan(obj):
                return 'CUDART_NAN_F'
            elif numpy.isinf(obj):
                if obj > 0:
                    return 'CUDART_INF_F'
                else:
                    return '-CUDART_INF_F'
            else:
                return str(float(obj)) + 'f'
        else:
            if numpy.isnan(obj):
                return 'CUDART_NAN'
            elif numpy.isinf(obj):
                if obj > 0:
                    return 'CUDART_INF'
                else:
                    return '-CUDART_INF'
            else:
                return str(float(obj))
    elif dtype.kind == 'c':
        if dtype.itemsize == 8:
            return f'thrust::complex<float>({obj.real}, {obj.imag})'
        elif dtype.itemsize == 16:
            return f'thrust::complex<double>({obj.real}, {obj.imag})'
    elif dtype.name == "bfloat16":
        # NOTE(seberg): It would be nice to find a more extensible path here.
        float_repr = _get_cuda_scalar_repr(obj, numpy.dtype(numpy.float32))
        return f"bfloat16({float_repr})"

    raise TypeError(f'Unsupported dtype: {dtype}')
