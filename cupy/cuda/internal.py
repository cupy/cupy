import ctypes
import platform
import sys


_support_cuda_versions = (75, 70, 65)


def load_library(names):
    if isinstance(names, str):
        names = [names]
    if 'linux' in sys.platform:
        template = 'lib%s.so'
        module = ctypes.cdll
    elif 'darwin' == sys.platform:
        template = 'lib%s.dylib'
        module = ctypes.cdll
    elif 'win32' == sys.platform:
        template = '%s.dll'
        module = ctypes.windll
    else:
        raise RuntimeError('Unsupported platform: %s' % sys.platform)

    names = [template % i for i in names]
    for name in names:
        try:
            return module.LoadLibrary(name)
        except OSError:
            pass
    else:
        raise OSError('library not found. %s' % names)


def get_windows_cuda_library_names(name):
    if platform.machine().endswith('64'):
        bit = 64
    else:
        bit = 32
    return ['%s%s_%s' % (name, bit, ver) for ver in _support_cuda_versions]
