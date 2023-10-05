# distutils: language = c++

"""Wrapper of Jitify utilities for CuPy API."""

from libcpp cimport nullptr
from libcpp.map cimport map as cpp_map
from libcpp.string cimport string as cpp_str
from libcpp.vector cimport vector

from cupy.cuda import cub

import atexit
import os
import pickle
import tempfile


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


# We need an internal way to invalidate the cache (say, when cuda_workaround.h
# or the CCCL bundle is updated) without having to set the environment variable
# CUPY_DISABLE_JITIFY_CACHE in the CI. This should never be touched by end
# users.
cdef extern from *:
    """
    const int build_num = 1;
    """
    const int build_num


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
_jitify_cache_dir = None
_jitify_cache_versions = None


cpdef _add_sources(dict sources):
    for hdr_name, hdr_source in sources.items():
        cupy_headers[hdr_name] = hdr_source


@atexit.register
def dump_cache():
    # Set up a version guard for invalidating the cache. Right now,
    # we use the build-time versions of CUB/Jitify.
    # TODO(leofang): Parse CUB/Thrust/libcu++ versions at process-
    # start time, for enabling CCCL + CuPy developers?
    assert _jitify_cache_versions is not None
    data = (_jitify_cache_versions, dict(cupy_headers))

    # Ensure the directory exists
    os.makedirs(_jitify_cache_dir, exist_ok=True)

    # Set up a temporary file; it must be under the cache directory so
    # that atomic moves within the same filesystem can be guaranteed
    with tempfile.NamedTemporaryFile(
            dir=_jitify_cache_dir, delete=False) as f:
        pickle.dump(data, f)
        f_name = f.name

    # atomic move with the destination guaranteed to be overwritten
    os.replace(f_name, f'{_jitify_cache_dir}/jitify.pickle')


cdef inline void _init_cupy_headers_from_cache() except*:
    global _jitify_cache_dir
    _jitify_cache_dir = os.getenv(
        'CUPY_CACHE_DIR', os.path.expanduser('~/.cupy/jitify_cache'))
    global _jitify_cache_versions
    versions = f"{get_build_version()}_{cub.get_build_version()}_{build_num}"
    _jitify_cache_versions = versions

    with open(f'{_jitify_cache_dir}/jitify.pickle', 'rb') as f:
        data = pickle.load(f)

    # Any failing sanity check here would mean the cache is invalidated.
    assert isinstance(data, tuple)
    assert len(data) == 2
    cached_versions, cached_headers = data
    assert isinstance(cached_versions, str)
    assert isinstance(cached_headers, dict)
    # Check the version guard for invalidating the cache (see the comment
    # in the dump_cache() function).
    assert cached_versions == versions

    # Populate the in-memory cache with the disk/persistent cache
    _add_sources(cached_headers)


cdef inline void _init_cupy_headers_from_scratch() except*:
    cdef int i
    for i in range(preinclude_jitsafe_headers_count):
        hdr_name = preinclude_jitsafe_header_names[i]
        hdr_source = get_jitsafe_headers_map().at(hdr_name)
        cupy_headers[hdr_name] = hdr_source

    # WAR
    # Jitify's type_traits is problematic with recent CCCL. Here, we populate
    # the cache with a dummy header source, so that Jitify's version wouldn't
    # be used. cupy/cuda_workaround.h is the real place importing type_traits
    # (from libcudacxx).
    cupy_headers[b"type_traits"] = b"#include <cupy/cuda_workaround.h>\n"
    # Same for tuple
    cupy_headers[b"tuple"] = b"#include <cupy/cuda_workaround.h>\n"


cdef inline void init_cupy_headers() except*:
    if int(os.getenv('CUPY_DISABLE_JITIFY_CACHE', '0')) == 0:
        try:
            _init_cupy_headers_from_cache()
        except Exception:
            pass  # continue to the old logic below
        else:
            return
    _init_cupy_headers_from_scratch()


init_cupy_headers()


# Use Jitify's internal mechanism to search all included headers, and return
# the modified options and the header mapping (as two lists). This roughly
# follows the constructor of jitify::Program(). The found headers are cached
# to accelerate Jitify's search loop.
cpdef jitify(str code, tuple opt):

    # input
    cdef cpp_str cuda_source
    cdef vector[cpp_str] headers

    # output
    cdef vector[cpp_str] include_paths
    cdef cpp_map[cpp_str, cpp_str] _sources = cupy_headers  # copy
    cdef vector[cpp_str] _options  # the input gets modified
    cdef cpp_str _name
    cdef list new_opt = None
    cdef list hdr_codes = []
    cdef list hdr_names = []

    # dummy
    cdef cpp_str h
    cdef str s
    cdef bytes k, v

    cuda_source = code.encode()
    _options = [s.encode() for s in opt]

    with nogil:
        # Where the real magic happens: a compile-fail-search loop
        load_program(cuda_source, headers, nullptr, &include_paths,
                     &_sources, &_options, &_name)

        # Remove input code from header cache
        _sources.erase(_name)

    # Get updated options. The two additions are from
    # jitify::detail::compile_kernel()
    new_opt = [h.decode() for h in _options]
    new_opt += ['--device-as-default-execution-space',
                '--pre-include=jitify_preinclude.h']

    # Collect header names and contents (as bytes)
    for itr in _sources:  # itr is an iterator of std::map
        k = itr.first
        v = itr.second
        hdr_codes.append(v)
        hdr_names.append(k)
        cupy_headers[k] = v

    return _name.decode(), tuple(new_opt), hdr_codes, hdr_names
