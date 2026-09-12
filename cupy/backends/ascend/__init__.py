"""Ascend (CANN) backend package.

This module performs a lightweight, import-time consistency check between the
CANN version the wheel was **built** against and the CANN version actually
**installed** on the machine.

Why this is needed
------------------
The aclnn operator ABI is not stable across CANN releases (operator signatures
change), and CANN couples ``libopapi``/``libop_common`` to a matching
``liboptiling.so``. A binary compiled against CANN 8.5 therefore can not be
expected to load against a 9.0 install. Without this check the failure mode is
an opaque ``undefined symbol`` / ``EL0003`` error deep inside aclnn, or a
segfault; with it the user gets an actionable message.

The check is intentionally *advisory*: it warns rather than raising, because
patch-level differences within a release train are usually fine and a false
positive must not make the package unimportable. Set
``CUPY_ASCEND_SKIP_VERSION_CHECK=1`` to silence it entirely.
"""

from __future__ import annotations

import json
import os
import re

__all__ = ['check_cann_version', 'get_wheel_metadata']

#: Matches the first ``major.minor.patch`` occurrence, as the build system does.
_VERSION_RE = re.compile(r'(\d+)\.(\d+)\.(\d+)')

#: Path of the metadata file written by the build system.
_WHEEL_JSON = os.path.join(os.path.dirname(__file__), '..', '..', '.data',
                           '_wheel.json')

_CHECKED = False


def get_wheel_metadata() -> dict | None:
    """Return the build-time metadata recorded in ``cupy/.data/_wheel.json``.

    Returns ``None`` when the file is absent (source build without the wheel
    metadata step, or a build predating this mechanism).
    """
    path = os.path.abspath(_WHEEL_JSON)
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _detect_installed_version() -> int | None:
    """Best-effort detection of the runtime CANN version.

    Reads the same version file the build system uses, under
    ``$ASCEND_HOME_PATH``. Returns ``None`` when it can not be determined, in
    which case the check is skipped rather than producing a spurious warning.
    """
    cann_path = os.environ.get('ASCEND_HOME_PATH')
    if not cann_path:
        return None

    # Keep this list in sync with `_find_cann_version_file()` in
    # `install/cupy_builder/install_build.py`: the file name/layout changed
    # across CANN releases.
    #   <cann>/version.cfg            - CANN 8.2 and earlier
    #   <cann>/compiler/version.info  - CANN 8.5+ (toolkit)
    #   <cann>/opp/version.info       - CANN 8.5+ (opp, fallback)
    candidates = [
        os.path.join(cann_path, 'version.cfg'),
        os.path.join(cann_path, 'compiler', 'version.info'),
        os.path.join(cann_path, 'opp', 'version.info'),
    ]
    for path in candidates:
        version = _parse_version_file(path)
        if version is not None:
            return version
    return None


def _parse_version_file(path: str) -> int | None:
    """Parse the CANN version out of a version file.

    Mirrors the build-time logic in ``install_build._find_cann_version_file``:
    a regex over the whole file, which handles both
    ``version=[8.2.0.0.0.201:xxx]`` (CANN 8.2) and plain ``Version=8.5.1``
    (CANN 8.5+).
    """
    try:
        with open(path) as f:
            content = f.read()
    except OSError:
        return None

    match = _VERSION_RE.search(content)
    if not match:
        return None
    major, minor, patch = (int(g) for g in match.groups())
    return major * 100 + minor * 10 + patch


def _format(version: int | None) -> str:
    if version is None or version < 0:
        return 'unknown'
    major, rest = divmod(version, 100)
    minor, patch = divmod(rest, 10)
    return f'{major}.{minor}.{patch}'


def check_cann_version(*, warn: bool = True) -> bool:
    """Compare the build-time and runtime CANN versions.

    Returns ``True`` when they are compatible (or compatibility could not be
    determined), ``False`` when a mismatch was detected. Emits a warning unless
    ``warn`` is false or ``CUPY_ASCEND_SKIP_VERSION_CHECK`` is set.
    """
    global _CHECKED
    _CHECKED = True

    if os.environ.get('CUPY_ASCEND_SKIP_VERSION_CHECK'):
        return True

    metadata = get_wheel_metadata()
    if not metadata:
        # Source build: the compile-time macro already matched at build time.
        return True

    built = metadata.get('cann_version')
    if not isinstance(built, int) or built < 0:
        return True

    installed = _detect_installed_version()
    if installed is None:
        return True

    # Compare on the release train (major.minor): patch releases within the
    # same train keep the aclnn ABI.
    if built // 10 == installed // 10:
        return True

    if warn:
        import warnings
        warnings.warn(
            'The installed CANN version ({installed}) differs from the '
            'version this CuPy build was compiled against ({built}); the '
            'aclnn operator ABI is not guaranteed to be compatible across '
            'releases. Reinstall a matching wheel (the wheel platform tag '
            'encodes the CANN version) or rebuild from source with '
            'CUPY_INSTALL_USE_ASCEND=1. Set '
            'CUPY_ASCEND_SKIP_VERSION_CHECK=1 to silence this warning.'.format(
                installed=_format(installed), built=_format(built)),
            RuntimeWarning,
            stacklevel=2,
        )
    return False


def checked() -> bool:
    """Whether :func:`check_cann_version` has already run."""
    return _CHECKED


# Run once at import; must not raise, as a false positive would make the
# package unimportable.
try:
    check_cann_version()
except Exception:  # noqa: BLE001 - never break import
    pass
