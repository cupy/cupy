# distutils: language = c++

"""Wrapper of CUB functions for CuPy API."""

from cpython cimport sequence
from libcpp.map cimport map as cpp_map 
from libcpp.string cimport string as cpp_str
from libcpp.vector cimport vector

from cupy_backends.cuda.api.driver cimport Stream as Stream_t
from cupy_backends.cuda.api cimport runtime
from cupy.cuda cimport common
from cupy.cuda cimport device
from cupy.cuda cimport memory
from cupy.cuda cimport stream


###############################################################################
# Extern
###############################################################################

cdef extern from '../core/include/cupy/jitify/jitify.hpp' namespace "jitify::detail" nogil:
    cpp_map[cpp_str, cpp_str]& get_jitsafe_headers_map()
    bint load_source(cpp_str, cpp_map[cpp_str, cpp_str]&, cpp_str, vector[cpp_str], void*, cpp_map[cpp_str, cpp_str]*, bint)
    const int preinclude_jitsafe_headers_count
    const char* preinclude_jitsafe_header_names[]


# Use Jitify's mechanism to fix all header includes, and return the
# modified source code. This roughly follows jitify::details::load_program().
cpdef jitify_source(str code, tuple opt, dict cached_sources=None) except +:
    cdef vector[cpp_str] include_paths, headers
    cdef cpp_str cuda_source = code.encode()
    cdef cpp_str hdr_source
    cdef cpp_map[cpp_str, cpp_str] sources
    cdef const char* hdr_name

    cdef list options = list(opt)
    cdef str s
    cdef bint result
    cdef int i

    if cached_sources is not None:
        for k, v in cached_sources.items():
            sources[k.encode()] = v.encode()

    # Add pre-include built-in JIT-safe headers
    for i in range(preinclude_jitsafe_headers_count):
        hdr_name = preinclude_jitsafe_header_names[i]
        hdr_source = get_jitsafe_headers_map().at(hdr_name)
        s = str(hdr_name) + "\n" + str(hdr_source)
        headers.push_back(s.encode())

    # Extract include paths from compile options
    for s in options:
        if s.startswith('-I'):
            include_paths.push_back(s[2:].encode())
            options.remove(s)

    # Load program source
    with nogil:
        result = load_source(cuda_source, sources, cpp_str(), include_paths, NULL, NULL, True)
    print(result)
    if not result:
        raise RuntimeError
    return str(cuda_source), tuple(options)


cpdef get_jit_map():
    cdef dict m = get_jitsafe_headers_map()

    return m


cpdef jitify_load_source():
    cdef cpp_str a, c
    cdef cpp_map[cpp_str, cpp_str] b, e
    cdef vector[cpp_str] d
    cdef bint f = 0
    cdef bint out

    with nogil:
        out = load_source(a, b, c, d, NULL, &e, f)
    return out
