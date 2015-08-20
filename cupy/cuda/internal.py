import ctypes
import sys


def load_library(name):
    if 'linux' in sys.platform:
        libname = 'lib%s.so' % name
        module = ctypes.cdll
    elif 'darwin' == sys.platform:
        libname = 'lib%s.dylib' % name
        module = ctypes.cdll
    elif 'windows' == sys.platform:
        libname = '%s.dll' % name
        module = ctypes.windll
    else:
        raise RuntimeError('Unsupported platform: %s' % sys.platform)

    return module.LoadLibrary(libname)
