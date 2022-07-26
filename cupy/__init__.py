from cupy import _environment
from cupy import _version


__version__ = _version.__version__

_environment._detect_duplicate_installation()  # NOQA
_environment._setup_win32_dll_directory()  # NOQA
_environment._preload_library('cutensor')  # NOQA
_environment._preload_library('nccl')  # NOQA


from cupy._from_numpy import *  # NOQA
from cupy._from_numpy import __getattr__  # NOQA
from cupy._stub import *  # NOQA


try:
    _import_error = None
    from cupy._init import *  # NOQA
except Exception as exc:
    _import_error = exc

    def raise_import_failure() -> None:
        raise RuntimeError(f'''
================================================================
{_environment._diagnose_import_error()}

Original error:
  {type(_import_error).__name__}: {_import_error}
================================================================
''') from _import_error

    def __getattr__(key: str) -> None:  # NOQA
        raise_import_failure()
