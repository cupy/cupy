# =============================================================================
# compile_with_cache, split from core.pyx
# =============================================================================
cimport cython  # NOQA
cimport cpython

from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda.libs cimport nvrtc

cdef bint _is_hip = runtime._is_hip_environment
cdef str _cuda_path = ''  # '' for uninitialized, None for non-existing
cdef str _bundled_include = ''  # '' for uninitialized, None for non-existing
_headers_from_wheel_available = False

cdef list cupy_header_list = [
    'cupy/complex.cuh',
    'cupy/carray.cuh',
    'cupy/atomics.cuh',
    'cupy/math_constants.h',
]
if _is_hip:
    cupy_header_list.append('cupy/hip_workaround.cuh')

# expose to Python for unit testing
_cupy_header_list = cupy_header_list

cdef str _cupy_header = ''.join(
    ['#include <%s>\n' % i for i in cupy_header_list])

# This is indirect include header list.
# These header files are subject to a hash key.
cdef list _cupy_extra_header_list = [
    'cupy/complex/complex.h',
    'cupy/complex/math_private.h',
    'cupy/complex/complex_inl.h',
    'cupy/complex/arithmetic.h',
    'cupy/complex/cproj.h',
    'cupy/complex/cexp.h',
    'cupy/complex/cexpf.h',
    'cupy/complex/clog.h',
    'cupy/complex/clogf.h',
    'cupy/complex/cpow.h',
    'cupy/complex/ccosh.h',
    'cupy/complex/ccoshf.h',
    'cupy/complex/csinh.h',
    'cupy/complex/csinhf.h',
    'cupy/complex/ctanh.h',
    'cupy/complex/ctanhf.h',
    'cupy/complex/csqrt.h',
    'cupy/complex/csqrtf.h',
    'cupy/complex/catrig.h',
    'cupy/complex/catrigf.h',
]

cdef str _header_path_cache = None
cdef str _header_source = None
cdef dict _header_source_map = {}


cpdef str _get_header_dir_path():
    global _header_path_cache
    if _header_path_cache is None:
        # Cython cannot use __file__ in global scope
        _header_path_cache = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'include'))
    return _header_path_cache


cpdef tuple _get_cccl_include_options():
    # the search paths are made such that they resemble the layout in CTK
    return (f"-I{_get_header_dir_path()}/cupy/_cccl/cub",
            f"-I{_get_header_dir_path()}/cupy/_cccl/thrust",
            f"-I{_get_header_dir_path()}/cupy/_cccl/libcudacxx")


cpdef str _get_header_source():
    global _header_source
    global _header_source_map
    cdef str header_path, base_path, file_path, header
    cdef list source

    if _header_source is None or not _header_source_map:
        source = []
        base_path = _get_header_dir_path()
        for file_path in _cupy_extra_header_list + cupy_header_list:
            header_path = os.path.join(base_path, file_path)
            with open(header_path) as header_file:
                header = header_file.read()
            source.append(header)
            _header_source_map[file_path.encode()] = header.encode()
        _header_source = '\n'.join(source)
    return _header_source


cpdef dict _get_header_source_map():
    global _header_source_map
    if not _header_source_map:
        _get_header_source()
    return _header_source_map


# added at the module level for precompiling the regex
_cucomplex_include_tokens = ['', '#', 'include', '<', r'cuComplex\.h', '>']
_cucomplex_include_pattern = re.compile(r'\s*'.join(_cucomplex_include_tokens))


cdef inline str _translate_cucomplex_to_thrust(str source):
    lines = []
    for line in source.splitlines(keepends=True):
        if _cucomplex_include_pattern.match(line):
            lines += '#include <cupy/cuComplex_bridge.h>  '\
                     '// translate_cucomplex\n'
        else:
            lines += line
    return ''.join(lines)


cpdef bint use_default_std(tuple options):
    cdef str opt
    for opt in options:
        if ('-std=' in opt) or ('--std=' in opt):
            return False
    return True

cpdef void warn_on_unsupported_std(tuple options):
    cdef str opt
    for opt in options:
        if _is_hip:
            major, minor = nvrtc.getVersion()
            if '-std=c++11' in opt and minor == 0 and major == 7:
                warnings.warn(
                    'hipRTC on some ROCm 7.0 builds have a known bug '
                    'that causes RTC to break when the standard is set '
                    'to c++11. Please use c++14, or use ROCm > 7.0'
                )
        elif '-std=c++03' in opt:
            warnings.warn('CCCL requires c++11 or above')


cpdef tuple assemble_cupy_compiler_options(tuple options):
    if use_default_std(options):
        options += ('--std=c++14',)
    else:
        warn_on_unsupported_std(options)

    # make sure bundled CCCL is searched first
    options = (_get_cccl_include_options()
               + options
               + ('-I%s' % _get_header_dir_path(),))

    global _cuda_path
    if _cuda_path == '':
        if not _is_hip:
            _cuda_path = cuda.get_cuda_path()
        else:
            _cuda_path = cuda.get_rocm_path()

    if not _is_hip:
        # CUDA Enhanced Compatibility
        global _bundled_include, _headers_from_wheel_available
        if _bundled_include == '':
            major, minor = nvrtc.getVersion()
            if major == 11:
                _bundled_include = 'cuda-11'
            elif major == 12 and minor < 2:
                # Use bundled header for CUDA 12.0 and 12.1 only.
                _bundled_include = 'cuda-12'
            else:
                # Do not use bundled includes dir after CUDA 12.2+.
                _bundled_include = None

            # Check if headers from cudart wheels are available.
            wheel_dir_count = len(
                _environment._get_include_dir_from_conda_or_wheel(
                    major, minor))
            _headers_from_wheel_available = (0 < wheel_dir_count)

        if (_bundled_include is None and
                _cuda_path is None and
                not _headers_from_wheel_available):
            raise RuntimeError(
                'Failed to auto-detect CUDA root directory. '
                'Please specify `CUDA_PATH` environment variable if you '
                'are using CUDA versions not yet supported by CuPy.')

        if _bundled_include is not None:
            options += ('-I' + os.path.join(
                _get_header_dir_path(), 'cupy', '_cuda', _bundled_include),)
    elif _is_hip:
        if _cuda_path is None:
            raise RuntimeError(
                'Failed to auto-detect ROCm root directory. '
                'Please specify `ROCM_HOME` environment variable.')

    if _cuda_path is not None:
        options += ('-I' + os.path.join(_cuda_path, 'include'),)

    return options


cpdef function.Module compile_with_cache(
        str source, tuple options=(), arch=None, cachd_dir=None,
        prepend_cupy_headers=True, backend='nvrtc', translate_cucomplex=False,
        enable_cooperative_groups=False, name_expressions=None,
        log_stream=None, bint jitify=False):
    if translate_cucomplex:
        source = _translate_cucomplex_to_thrust(source)
        cupy_header_list.append('cupy/cuComplex_bridge.h')
        prepend_cupy_headers = True

    if prepend_cupy_headers:
        source = _cupy_header + source
    if jitify:
        source = '#include <cupy/cuda_workaround.h>\n' + source
    extra_source = _get_header_source()

    options = assemble_cupy_compiler_options(options)

    return cuda.compiler._compile_module_with_cache(
        source, options, arch, cachd_dir, extra_source, backend,
        enable_cooperative_groups=enable_cooperative_groups,
        name_expressions=name_expressions, log_stream=log_stream,
        jitify=jitify)
