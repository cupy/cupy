# distutils: language = c++

"""Wrapper of Jitify utilities for CuPy API."""

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

# Use Jitify's mechanism to fix all header includes, and return the modified
# options and the header mapping. This roughly follows the constructor of
# jitify::Program.
cpdef jitify(str code, tuple opt, dict cached_sources=None):
    """ 

    .. note::
        cached_sources is a map from raw input codes to the transformed codes

    """

    # input
    cdef cpp_str cuda_source
    cdef vector[cpp_str] headers, include_paths

    # output
    cdef cpp_map[cpp_str, cpp_str] _sources
    cdef vector[cpp_str] _options
    cdef cpp_str _name
    cdef list new_opt = []

    # dummy
    cdef cpp_str hdr_name, hdr_source, h
    cdef str s, k, v
    cdef int i

    # Populate the C++ map using Python map
    # TODO: I think _sources should always have a fresh start!
    if cached_sources is not None:
        for k, v in cached_sources.items():
            _sources[k.encode()] = v.encode()

    cuda_source = code.encode()
    _options = [s.encode() for s in opt]

    # Add pre-include built-in JIT-safe headers
    for i in range(preinclude_jitsafe_headers_count):
        hdr_name = preinclude_jitsafe_header_names[i]
        hdr_source = get_jitsafe_headers_map().at(hdr_name)
        hdr_name += <cpp_str>(b"\n") + hdr_source
        #print(s)
        headers.push_back(hdr_name)

    # call add_options_from_env()?
    
    with nogil:
        load_program(cuda_source, headers, NULL, &include_paths,
                     &_sources, &_options, &_name)

    # Update the Python map using C++ map
    # TODO: just store them as two lists?
    if cached_sources is not None:
        for kv in _sources:  # kv is std::pair<string, string>
            cached_sources[kv.first] = kv.second  # store as bytes

    # see jitify::detail::compile_kernel()
    _options.push_back(b'--device-as-default-execution-space')
    _options.push_back(b'--pre-include=jitify_preinclude.h')
    for h in _options:
        new_opt.append(h.decode())

    return _name.decode(), tuple(new_opt), cached_sources
