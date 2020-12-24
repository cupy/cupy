# distutils: language = c++

"""Wrapper of Jitify utilities for CuPy API."""

from cython.operator cimport dereference as deref
from libcpp cimport nullptr
from libcpp.map cimport map as cpp_map
from libcpp.string cimport string as cpp_str
from libcpp.vector cimport vector


###############################################################################
# Extern
###############################################################################

cdef extern from 'cupy_jitify.h' namespace "jitify::detail" nogil:
    cpp_map[cpp_str, cpp_str]& get_jitsafe_headers_map()
    const int preinclude_jitsafe_headers_count
    const char* preinclude_jitsafe_header_names[]
    void load_program(cpp_str&,
                      vector[cpp_str]&,
                      void*,
                      vector[cpp_str]*,
                      cpp_map[cpp_str, cpp_str]*,
                      vector[cpp_str]*,
                      cpp_str*) except +

    const char* jitify_ver  # set at build time


###############################################################################
# API
###############################################################################

def get_build_version():
    if jitify_ver == b'-1':
        return '<unknown>'
    return jitify_ver.decode()


# We cache headers found by Jitify. This is initialized with a few built-in
# JIT-safe headers, and expands as needed to help reduce compile time.
cdef cpp_map[cpp_str, cpp_str] cupy_headers
cdef inline void init_cupy_headers():
    cdef int i
    for i in range(preinclude_jitsafe_headers_count):
        hdr_name = preinclude_jitsafe_header_names[i]
        hdr_source = get_jitsafe_headers_map().at(hdr_name)
        cupy_headers[hdr_name] = hdr_source


init_cupy_headers()


# Use Jitify's internal mechanism to search all included headers, and return
# the modified options and the header mapping (as two lists). This roughly
# follows the constructor of jitify::Program(). The found headers are cached
# to accelerate Jitify's search loop.
cpdef jitify(str code, tuple opt, dict cached_sources=None):

    # input
    cdef cpp_str cuda_source
    cdef vector[cpp_str] headers

    # output
    cdef vector[cpp_str] include_paths
    cdef cpp_map[cpp_str, cpp_str]* _sources = &cupy_headers
    cdef vector[cpp_str] _options  # the input gets modified
    cdef cpp_str _name
    cdef list new_opt = None
    cdef list hdr_codes = []
    cdef list hdr_names = []

    # dummy
    cdef cpp_str hdr_name, hdr_source, h
    cdef str s
    cdef bytes k, v
    cdef int i

    cuda_source = code.encode()
    _options = [s.encode() for s in opt]

    # Populate the cpp map
    if cached_sources is not None:
        for k, v in cached_sources.items():
            hdr_name = k
            hdr_source = v
            cupy_headers[hdr_name] = hdr_source

    with nogil:
        # Where the real magic happens: a compile-fail-search loop
        load_program(cuda_source, headers, nullptr, &include_paths,
                     _sources, &_options, &_name)

        # Remove input code from header cache
        _sources.erase(_name)

    # Get updated options. The two additions are from
    # jitify::detail::compile_kernel()
    new_opt = [h.decode() for h in _options]
    new_opt += ['--device-as-default-execution-space',
                '--pre-include=jitify_preinclude.h']

    # Collect header names and contents (as bytes)
    for itr in deref(_sources):  # itr is an iterator of std::map
        k = itr.first
        v = itr.second
        hdr_codes.append(v)
        hdr_names.append(k)

    return _name.decode(), tuple(new_opt), hdr_codes, hdr_names
