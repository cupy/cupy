"""Abstract ``Backend`` interface for the CuPy build system.

A ``Backend`` encapsulates *everything the build system needs to know about one
accelerator platform*:

* where its SDK lives (``get_sdk_path``);
* which device compiler to invoke (``get_device_compiler``);
* include/library search paths (``get_include_dirs`` / ``get_library_dirs``);
* compiler options and preprocessor macros
  (``get_extra_compile_args`` / ``get_define_macros``);
* the device-compilation command line (``get_device_compile_args``);
* the Cython compile-time constants (``get_compile_time_env``);
* version detection (``check_version`` / ``get_version``).

This keeps all backend-specific knowledge inside a single module per backend
instead of being spread as ``if backend == ...`` chains across
``install_build.py`` / ``_compiler.py`` / ``_command.py`` / ``cupy_setup_build.py``.

See ``install/README.md`` for how to add a new backend.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cupy_builder._context import Context


class Backend(abc.ABC):
    """Abstract base class describing one accelerator backend."""

    #: Sentinel used when a version could not be determined.
    NOT_AVAILABLE = -1

    #: Canonical short name, e.g. ``'cuda'``, ``'rocm'``, ``'ascend'``.
    name: str = ''

    def __init__(self) -> None:
        self._version_cache: int = self.NOT_AVAILABLE

    #: Environment variable that turns this backend on during install,
    #: e.g. ``'CUPY_INSTALL_USE_ASCEND'``. Empty for the default backend.
    env_flag: str = ''

    #: Name of the Cython compile-time constant holding this backend's
    #: version, e.g. ``'CUPY_CANN_VERSION'``.
    version_macro: str = ''

    #: Environment variable holding the SDK root path (used for the
    #: configuration summary), e.g. ``'ASCEND_HOME_PATH'``.
    sdk_env_var: str = ''

    #: Environment variable that can override the device compiler,
    #: e.g. ``'NVCC'`` / ``'HIPCC'``. Empty when unsupported.
    compiler_env_var: str = ''

    #: Human-readable name of the device compiler, used in messages.
    compiler_name: str = ''

    #: Whether this backend needs CUB/Thrust/libcudacxx headers on the
    #: include path (CUDA and ROCm do; Ascend does not).
    needs_cub_headers: bool = False

    # ------------------------------------------------------------------
    # SDK / compiler discovery
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def get_sdk_path(self) -> str | None:
        """Return the SDK root directory, or ``None`` if not found."""

    @abc.abstractmethod
    def get_device_compiler(self) -> list[str] | None:
        """Return the device compiler command as an argv list, or ``None``."""

    # ------------------------------------------------------------------
    # Build settings
    # ------------------------------------------------------------------
    def get_include_dirs(self, ctx: Context) -> list[str]:
        """Return backend-specific include directories."""
        return []

    def get_library_dirs(self, ctx: Context) -> list[str]:
        """Return backend-specific library directories."""
        return []

    def get_extra_compile_args(self) -> list[str]:
        """Return backend-specific extra compile flags (host compiler)."""
        return []

    def get_extra_link_args(self) -> list[str]:
        """Return backend-specific extra link flags."""
        return []

    # ------------------------------------------------------------------
    # RPATH policy
    # ------------------------------------------------------------------
    #: Whether the SDK's ``library_dirs`` may be embedded into the built
    #: extension modules' RPATH.
    #:
    #: ``True``  -- the SDK is normally a system install that will still be
    #:              present at runtime (CUDA/ROCm under ``/usr/local``).
    #: ``False`` -- the SDK is a relocatable, user-installed toolkit whose
    #:              absolute path must never leak into a redistributable
    #:              binary (Ascend/CANN).
    #:
    #: When ``False``, ``$ORIGIN``-relative entries are still emitted so that
    #: libraries bundled under ``cupy/.data/lib`` remain discoverable.
    embed_sdk_in_rpath: bool = True

    def get_wheel_platform_tag(self) -> str | None:
        """Return a platform-tag suffix identifying the SDK, or ``None``.

        The value is appended to the wheel's platform tag, e.g. ``'cann8.5'``
        producing ``manylinux_2_17_x86_64.cann8.5``. This lets several
        mutually-incompatible SDK builds coexist on an index without being
        mistaken for one another.
        """
        return None

    def get_wheel_metadata(self) -> dict[str, Any]:
        """Return extra entries for ``cupy/.data/_wheel.json``.

        These are recorded in the wheel and checked again at runtime, so a
        binary built against one SDK version refuses to load against an
        incompatible one with a clear error instead of crashing.
        """
        return {}

    def get_define_macros(self) -> list[tuple[str, str]]:
        """Return backend-specific preprocessor macros."""
        return []

    # ------------------------------------------------------------------
    # Device compilation
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def get_device_compile_args(self, ctx: Context, src: str) -> list[str]:
        """Return the compiler invocation (argv) for one device source file.

        The returned list must already contain the compiler executable and all
        flags except the source/output arguments, which the caller appends.
        """

    def supports_platform(self, platform: str) -> bool:
        """Whether this backend can build on ``platform`` (``'linux'``/``'win32'``)."""
        return platform == 'linux'

    # ------------------------------------------------------------------
    # Cython compile-time constants
    # ------------------------------------------------------------------
    def get_compile_time_env(self, ctx: Context) -> dict[str, Any]:
        """Return ``{macro_name: value}`` for Cython ``compile_time_env``."""
        return {}

    # ------------------------------------------------------------------
    # Version detection
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def check_version(self, compiler: Any, settings: Any) -> bool:
        """Detect and cache the backend version; return True when supported."""

    @abc.abstractmethod
    def get_version(self) -> int:
        """Return the cached backend version (see ``check_version``)."""
