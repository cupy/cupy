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
    # bint load_source(cpp_str, cpp_map[cpp_str, cpp_str]&, cpp_str, vector[cpp_str], void*, cpp_map[cpp_str, cpp_str]*, bint)
    const int preinclude_jitsafe_headers_count
    const char* preinclude_jitsafe_header_names[]
    void load_program(cpp_str&,
                      vector[cpp_str]&,
                      void*,
                      vector[cpp_str]*,
                      cpp_map[cpp_str, cpp_str]*,
                      vector[cpp_str]*,
                      cpp_str*) except +


# # Use Jitify's mechanism to fix all header includes, and return the
# # modified source code. This roughly follows jitify::details::load_program().
# cpdef jitify_source(str code, tuple opt, dict cached_sources=None) except +:
#     """ 
# 
#     .. note::
#         cached_sources is a map from raw input codes to the transformed codes
# 
#     """
# 
#     cdef vector[cpp_str] include_paths, headers
#     cdef cpp_str cuda_source = code.encode()
#     cdef cpp_str hdr_source
#     cdef cpp_str* program_name
#     cdef cpp_map[cpp_str, cpp_str] sources, header_fullpaths
#     cdef const char* hdr_name
# 
#     cdef list options = list(opt)
#     cdef str s, k, v
#     cdef bint result
#     cdef int i
# 
#     # Populate the C++ map using Python map
#     if cached_sources is not None:
#         for k, v in cached_sources.items():
#             sources[k.encode()] = v.encode()
# 
#     # Add pre-include built-in JIT-safe headers
#     for i in range(preinclude_jitsafe_headers_count):
#         hdr_name = preinclude_jitsafe_header_names[i]
#         hdr_source = get_jitsafe_headers_map().at(hdr_name)
#         s = str(hdr_name) + "\n" + str(hdr_source)
#         headers.push_back(s.encode())
# 
#     # Extract include paths from compile options
#     for s in options:
#         if s.startswith('-I'):
#             include_paths.push_back(s[2:].encode())
#             options.remove(s)
# 
#     # Load program source
#     with nogil:
#         result = load_source(cuda_source, sources, cpp_str(), include_paths, NULL, NULL, True)
#     if not result:
#         raise RuntimeError("Load source failed: " + cuda_source)
#     #*program_name = sources.begin().first
# 
#     # Load header sources
#     for h in headers:
#         with nogil:
#             result = load_source(h, sources, cpp_str(), include_paths, NULL, NULL, True)
#         if not result:
#             raise RuntimeError("Load source failed: " + cuda_source)
# 
#     # Update the Python map using C++ map
#     if cached_sources is not None:
#         for kv in sources:  # k is a C++ iterator!
#             cached_sources[kv.first.decode()] = kv.second.decode()
# 
#     return cuda_source.decode(), tuple(options), cached_sources


# Use Jitify's mechanism to fix all header includes, and return the
# modified source code. This roughly follows jitify::details::load_program().
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


# def check_result(int result, str msg):
#     if not result:
#         raise RuntimeError(msg)
# 
# 
# cpdef get_jit_map():
#     cdef dict m = get_jitsafe_headers_map()
# 
#     return m
# 
# 
# cpdef jitify_load_source():
#     cdef cpp_str a, c
#     cdef cpp_map[cpp_str, cpp_str] b, e
#     cdef vector[cpp_str] d
#     cdef bint f = 0
#     cdef bint out
# 
#     with nogil:
#         out = load_source(a, b, c, d, NULL, &e, f)
#     return out
