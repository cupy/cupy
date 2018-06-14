import ctypes
import ctypes.util
import os
import re
import sys
import warnings

import cupy.cuda.nvrtc


_nvrtc_platform_config = {
    'linux': {
        'nvrtc': 'libnvrtc.so',
        'nvrtc-builtins': 'libnvrtc-builtins.so',
        'lib_path_env': 'LD_LIBRARY_PATH',
        'version_func': lambda lib, base: base.split('{}.'.format(lib))[1],
    }
}
_nvrtc_platform_config['linux2'] = _nvrtc_platform_config['linux']


def _get_cdll_path(func):
    libdl_path = ctypes.util.find_library('dl')
    if libdl_path is None:
        return None

    try:
        libdl = ctypes.CDLL(libdl_path)
    except OSError:
        return None
    if not hasattr(libdl, 'dladdr'):
        return None

    class Dl_info(ctypes.Structure):
        _fields_ = (
            ('dli_fname', ctypes.c_char_p),
            ('dli_fbase', ctypes.c_void_p),
            ('dli_sname', ctypes.c_char_p),
            ('dli_saddr', ctypes.c_void_p),
        )

    libdl.dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(Dl_info))
    info = Dl_info()
    result = libdl.dladdr(func, ctypes.byref(info))
    if result == 0:
        return None

    return info.dli_fname.decode()


def _get_nvrtc_path():
    fp = cupy.cuda.nvrtc._get_function_pointer()
    path = _get_cdll_path(fp)
    return None if path is None else os.path.realpath(path)


def _get_nvrtc_builtins_path(libname):
    try:
        lib = ctypes.CDLL(libname)
    except OSError:
        return None
    if not hasattr(lib, 'getArchBuiltins'):
        return None
    path = _get_cdll_path(lib.getArchBuiltins)
    return None if path is None else os.path.realpath(path)


def check():
    conf = _nvrtc_platform_config.get(sys.platform, None)
    if conf is None:
        # Unsupported platform.
        return

    nvrtc_path = _get_nvrtc_path()
    nvrtc_builtins_path = _get_nvrtc_builtins_path(conf['nvrtc-builtins'])

    if nvrtc_builtins_path is None:
        if nvrtc_path is None:
            # libdl does not provide dladdr function.
            pass
        else:
            warnings.warn('''nvrtc-builtins ({}) could not be loaded.
Please make sure that all CUDA Toolkit components (including development \
libraries) are installed on your system.'''.format(conf['nvrtc-builtins']))
        return

    nvrtc_ver = conf['version_func'](
        conf['nvrtc'],
        os.path.basename(nvrtc_path))
    nvrtc_builtins_ver = conf['version_func'](
        conf['nvrtc-builtins'],
        os.path.basename(nvrtc_builtins_path))

    if nvrtc_ver == nvrtc_builtins_ver:
        return

    warnings.warn('''\
Version mismatch of nvrtc and nvrtc-builtins detected.

nvrtc           : {} (version {})
nvrtc-builtins  : {} (version {})

Please add {} to {} environment variable.
'''.format(
        nvrtc_path, nvrtc_ver,
        nvrtc_builtins_path, nvrtc_builtins_ver,
        os.path.dirname(nvrtc_path), conf['lib_path_env']))
