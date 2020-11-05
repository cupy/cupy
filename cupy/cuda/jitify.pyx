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


###############################################################################
# API
###############################################################################

cdef vector[cpp_str] jitsafe_headers = vector[cpp_str](0)


# Use Jitify's mechanism to fix all header includes, and return the modified
# options and the header mapping. This roughly follows the constructor of
# jitify::Program().
cpdef jitify(str code, tuple opt, dict cached_sources=None):

    # input
    cdef cpp_str cuda_source
    cdef vector[cpp_str]* headers = &jitsafe_headers

    # output
    cdef vector[cpp_str] include_paths
    cdef cpp_map[cpp_str, cpp_str] _sources
    cdef vector[cpp_str] _options  # the input gets modified
    cdef cpp_str _name
    cdef list new_opt = []

    # dummy
    cdef cpp_str hdr_name, hdr_source, h
    cdef str s
    cdef bytes k, v
    cdef int i

    # Unfortunately, Jitify does not allow us to reuse an existing header
    # mapping. The reason is when Jitify looks up the program name, it expects
    # the map only has 1 element, but if we pre-populate the map with the
    # cached elements, it would fail to identify the name (_name).
    # TODO(leofang): do we really need to know the correct name?
    if cached_sources is not None:
        raise ValueError('cached_sources should not be set')

    cuda_source = code.encode()
    _options = [s.encode() for s in opt]

    # Add pre-include built-in JIT-safe headers
    if headers.size() == 0:
        for i in range(preinclude_jitsafe_headers_count):
            hdr_name = preinclude_jitsafe_header_names[i]
            hdr_source = get_jitsafe_headers_map().at(hdr_name)
            hdr_name += <cpp_str>(b"\n") + hdr_source
            headers.push_back(hdr_name)

    # Where the real magic happens
    with nogil:
        load_program(cuda_source, deref(headers), nullptr, &include_paths,
                     &_sources, &_options, &_name)

    # Update the Python dict using C++ map
    if cached_sources is None:
        cached_sources = {}
    for k, v in _sources:  # (k, v) is std::pair<std::string, std::string>
        h = <cpp_str>(k)
        if h == _name:
            continue
        cached_sources[k] = v  # store as bytes

    # Get updated options. The two additions are from
    # jitify::detail::compile_kernel()
    _options.push_back(b'--device-as-default-execution-space')
    _options.push_back(b'--pre-include=jitify_preinclude.h')
    for h in _options:
        new_opt.append(h.decode())

    return _name.decode(), tuple(new_opt), cached_sources
