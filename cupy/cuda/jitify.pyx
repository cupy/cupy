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
cdef vector[cpp_str] cupy_headers = vector[cpp_str](0)
cdef dict cupy_headers_mapping = {}


# Use Jitify's mechanism to fix all header includes, and return the modified
# options and the header mapping. This roughly follows the constructor of
# jitify::Program().
cpdef jitify(str code, tuple opt, dict cached_sources=None):

    # input
    cdef cpp_str cuda_source
    cdef vector[cpp_str]* headers

    # output
    cdef vector[cpp_str] include_paths
    cdef cpp_map[cpp_str, cpp_str] _sources
    cdef vector[cpp_str] _options  # the input gets modified
    cdef cpp_str _name
    cdef list new_opt, hdr_codes, hdr_names

    # dummy
    cdef cpp_str hdr_name, hdr_source, h
    cdef str s
    cdef bytes k, v
    cdef int i

    cuda_source = code.encode()
    _options = [s.encode() for s in opt]

    # Add pre-include built-in JIT-safe headers
    global jitsafe_headers, cupy_headers
    if jitsafe_headers.size() == 0:
        for i in range(preinclude_jitsafe_headers_count):
            hdr_name = preinclude_jitsafe_header_names[i]
            hdr_source = get_jitsafe_headers_map().at(hdr_name)
            hdr_name += <cpp_str>(b"\n") + hdr_source
            jitsafe_headers.push_back(hdr_name)
        cupy_headers = jitsafe_headers  # copy construct

    # Unfortunately, Jitify does not allow us to reuse an existing header
    # mapping (_sources). The reason is when Jitify looks up the program name,
    # it expects the map only has 1 element, but if we pre-populate the map
    # with the cached elements, it would fail to identify the name (_name). As
    # a workaround, we follow the pre-include approach above for CuPy headers.
    # Assumption: cupy_headers_mapping is a subset of cached_sources, and in
    # every call of this function we pass in the latter (initialized elsewhere)
    # to update the former, such that the vector (cupy_headers) is gradually
    # expanded with no repeated elements.
    global cupy_headers_mapping
    if cached_sources is not None:
        for k, v in cached_sources.items():
            if k not in cupy_headers_mapping:
                hdr_name = k
                hdr_source = v
                cupy_headers_mapping[k] = v
                hdr_name += <cpp_str>(b"\n") + hdr_source
                cupy_headers.push_back(hdr_name)

    # Where the real magic happens
    headers = &cupy_headers
    with nogil:
        load_program(cuda_source, deref(headers), nullptr, &include_paths,
                     &_sources, &_options, &_name)

    # Get updated options. The two additions are from
    # jitify::detail::compile_kernel()
    _options.push_back(b'--device-as-default-execution-space')
    _options.push_back(b'--pre-include=jitify_preinclude.h')
    new_opt = [h.decode() for h in _options]

    # Update the Python dict using C++ map
    # Note: don't update cupy_headers_mapping; it's done in the next call
    # of this function!
    if cached_sources is None:
        cached_sources = {}
    for itr in _sources:  # itr is an iterator of std::map
        k_cpp = itr.first
        v_cpp = itr.second
        k = k_cpp
        if k_cpp == _name or k in cached_sources:
            continue
        v = v_cpp
        cached_sources[k] = v  # store as bytes

    # Although we already have cached_sources, for later convenience we
    # also split it into two matching lists
    hdr_codes = list(cached_sources.values())
    hdr_names = list(cached_sources.keys())

    return _name.decode(), tuple(new_opt), hdr_codes, hdr_names, cached_sources
