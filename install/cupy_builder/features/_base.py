"""Base ``Feature`` class shared by every backend.

A ``Feature`` describes one logical build unit: the Cython modules it
provides, the headers/libraries it needs and how to determine its version.
Backends (CUDA/ROCm/Ascend) either subclass :class:`Feature` or are described
by a plain dict turned into a ``Feature`` via :func:`from_dict`.
"""

from __future__ import annotations

from typing import Any

from cupy_builder import Context


class Feature:
    NOT_AVAILABLE = -1
    _UNDETERMINED = -100

    def __init__(self, ctx: Context):
        # Name of the feature.
        self.name = ''

        # When True, fail the build if the feature is unavailable.
        self.required = False

        # List of Cython modules.
        self.modules: list[str] = []

        # C/C++ headers required for the feature.
        # This is used only for testing availability of the feature.
        self.includes: list[str] = []

        # Libraries (shared/static) on the search path to be linked.
        self.libraries: list[str] = []

        # Static libraries (manually searched) to be linked.
        self.static_libraries: list[str] = []

        # Version of the feature.
        self._version: Any = self._UNDETERMINED

    def configure(self, compiler: Any, settings: Any) -> bool:
        # Fill `self._version` with the version or NOT_AVAILABLE
        self._version = None
        return True

    def get_version(self) -> Any:
        assert self._version != self._UNDETERMINED, 'not configured yet'
        return self._version

    def __contains__(self, key: Any) -> bool:
        # TODO(kmaehashi): Remove this transient function.
        if not isinstance(key, str):
            return False
        try:
            self.__getitem__(key)
        except AttributeError:
            return False
        return True

    def __getitem__(self, key: str) -> Any:
        # TODO(kmaehashi): Remove this transient function.
        if key == 'file':
            return self.modules
        elif key == 'include':
            return self.includes
        return getattr(self, key)


def from_dict(d: dict[str, Any], ctx: Context) -> Feature:
    """Define a feature from a plain dict.

    TODO(kmaehashi): Remove this transient function.
    """
    f = Feature(ctx)
    f.name = d['name']
    f.required = d.get('required', False)
    f.libraries = d['libraries']
    f.static_libraries = d.get('static_libraries', [])

    # Note: the following are renamed
    f.modules = d['file']
    f.includes = d['include']
    if 'check_method' in d:
        f.configure = d['check_method']  # type: ignore
        f._version = None
        if 'version_method' in d:
            f.get_version = d['version_method']  # type: ignore
    return f
