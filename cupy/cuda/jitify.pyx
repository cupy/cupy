# distutils: language = c++

"""Wrapper of Jitify utilities for CuPy API."""

from libcpp cimport nullptr
from libcpp.map cimport map as cpp_map
from libcpp.string cimport string as cpp_str
from libcpp.vector cimport vector

import json
import os
import re
import tempfile
import warnings

from cupy._environment import get_cuda_path
from cupy.cuda import cub
from cupy import _util


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
    const char* cupy_cache_key  # set at build time


# We need an internal way to invalidate the cache (when the warmup_kernel below
# is updated) without having to set the environment variable
# CUPY_DISABLE_JITIFY_CACHE in the CI. This should never be touched by end
# users.
cdef extern from *:
    """
    const int build_num = 3;
    """
    const int build_num


###############################################################################
# API
###############################################################################

def get_build_version():
    if jitify_ver == b'-1':
        return '<unknown>'
    return jitify_ver.decode()


cpdef str get_cuda_version():
    # Read CUDART version from header if it exists, otherwise use NVRTC version
    # as a proxy.
    cdef str cuda_path = get_cuda_path()
    cdef str cuda_ver = None

    if cuda_path is not None:
        try:
            with open(
                    os.path.join(cuda_path,
                                 'include/cuda_runtime_api.h')) as f:
                hdr = f.read()
            m = re.search(r'#define CUDART_VERSION\s+([0-9]*)', hdr)
            if m:
                cuda_ver = m.group(1)
        except:  # noqa:E722
            pass

    if cuda_ver is None:
        # avoid circular dependency
        from cupy.cuda.compiler import _get_nvrtc_version
        major, minor = _get_nvrtc_version()
        cuda_ver = f"{int(major) * 1000 + int(minor) * 10}"

    return cuda_ver


# We cache headers found by Jitify. This is initialized with a few built-in
# JIT-safe headers, and expands as needed to help reduce compile time.
cdef cpp_map[cpp_str, cpp_str] cupy_headers

# Module-level constants
cdef bint _jitify_init = False
cdef str _jitify_cache_dir = None
cdef str _jitify_cache_versions = None


cpdef _add_sources(dict sources, bint is_str=False):
    cdef str k, v
    for hdr_name, hdr_source in sources.items():
        if is_str:
            k = hdr_name
            v = hdr_source
            cupy_headers[k.encode()] = v.encode()
        else:  # name/source are raw bytes
            cupy_headers[hdr_name] = hdr_source


cdef inline void dump_cache(cpp_map[cpp_str, cpp_str]& cupy_headers) except*:
    # Ensure the directory exists
    os.makedirs(_jitify_cache_dir, exist_ok=True)

    # Construct a temporary Python dict for serialization
    cdef dict data = {}
    cdef bytes k, v
    for it in cupy_headers:
        k = it.first
        v = it.second
        data[k.decode()] = v.decode()

    # Set up a temporary file; it must be under the cache directory so
    # that atomic moves within the same filesystem can be guaranteed
    with tempfile.NamedTemporaryFile(
            mode='w', dir=_jitify_cache_dir, delete=False) as f:
        json.dump(data, f)
        f_name = f.name

    # atomic move with the destination guaranteed to be overwritten
    os.replace(f_name,
               f'{_jitify_cache_dir}/jitify_{_jitify_cache_versions}.json')


# This kernel simply includes commonly used headers in CuPy's codebase
# to populate the Jitify cache. Need to bump build_num if updated.
cdef str warmup_kernel = r"""cupy_jitify_exercise
#include <cupy/cuda_workaround.h>
#include <cuda_fp16.h>

#include <type_traits>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_load.cuh>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
// not supported before CC 7.0
#include <cuda/barrier>
#endif

extern "C" __global__ void jitify_exercise() { }
"""


cdef inline void _init_cupy_headers_from_cache() except*:
    # If this function raises an exception, it would mean the cache is
    # invalidated.
    assert _jitify_cache_versions is not None

    # Attempt to load from the disk/persistent cache
    cdef dict data
    with open(
            f'{_jitify_cache_dir}/jitify_{_jitify_cache_versions}.json',
            'r') as f:
        data = json.load(f)

    # Populate the cache (cupy_headers)
    _add_sources(data, is_str=True)

    global _jitify_init
    _jitify_init = True


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

    # Ensure users know this is normal and not hanging...
    warnings.warn(
        "Jitify is performing a one-time only warm-up to populate the "
        "persistent cache, this may take a few seconds and will be improved "
        "in a future release...", _util.PerformanceWarning)

    # Compile a dummy kernel to further populate the cache (with bundled
    # headers)
    # need to defer import to avoid circular dependency
    from cupy._core.core import assemble_cupy_compiler_options
    cdef tuple options = ('-std=c++11', '-DCUB_DISABLE_BF16_SUPPORT',)
    options = assemble_cupy_compiler_options(options)
    jitify(warmup_kernel, options)

    global _jitify_init
    _jitify_init = True

    # Frozen the cache (to not mix in user-provided headers)
    dump_cache(cupy_headers)


cdef inline void _init_cupy_headers() except*:
    if int(os.getenv('CUPY_DISABLE_JITIFY_CACHE', '0')) == 0:
        try:
            _init_cupy_headers_from_cache()
        except Exception:
            pass  # continue to the old logic below
        else:
            return
    _init_cupy_headers_from_scratch()


cpdef void _init_module() except*:
    if _jitify_init:
        return

    global _jitify_cache_dir
    if _jitify_cache_dir is None:
        _jitify_cache_dir = os.getenv(
            'CUPY_CACHE_DIR', os.path.expanduser('~/.cupy/jitify_cache'))

    # Set up a version guard for invalidating the cache. Right now, we use the
    # build-time versions of Jitify/CUB (CCCL)/CuPy and run-time versions of
    # CuPy and CUDA (cannot use cudaRuntimeGetVersion as we link statically to
    # it, but we still need a proxy for CTK).
    # TODO(leofang): Parse CUB/Thrust/libcu++ versions at process-
    # start time, for enabling CCCL + CuPy developers?
    global _jitify_cache_versions
    if _jitify_cache_versions is None:
        # jitify version could be "<unknown>" and the angular brakets are not
        # valid characters on Windows, so we need to strip them
        _jitify_version = re.sub(r'<([^>]*)>', r'\1', get_build_version())
        _jitify_cache_versions = (
            f"{_jitify_version}_{cub.get_build_version()}_"
            f"{get_cuda_version()}_{build_num}_{cupy_cache_key.decode()}")

    _init_cupy_headers()


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
