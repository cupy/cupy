"""``numpy_xpu`` -- the public import name of *numpy-ascend*.

The implementation lives in the ``cupy`` package: that tree is kept under its
original name because ~1M lines of Cython/C++/``.pxd`` sources and the compiled
extension module paths (``cupy._core.core``, ``cupy_backends...``) hard-code
``import cupy``.  Renaming the directory would mean editing every one of them.

This package therefore turns the two names into the *same module object*::

    import numpy_xpu as xp
    import cupy as cp
    assert xp is cp                  # works

It uses the same technique as the ``cupy_backends`` alias in
``cupy/__init__.py``: register the *real* module objects in ``sys.modules``
instead of importing the tree a second time under the new name.  Re-importing
would execute every module twice and produce two distinct class objects, which
breaks ``isinstance`` and ``except cupy.XxxError`` across the two names.

Notes
-----
* ``numpy_xpu.__name__`` is ``'cupy'`` and ``numpy_xpu.__file__`` points into
  the ``cupy`` tree.  That is intentional: it is one module, two names.
* Submodules are aliased lazily (``import numpy_xpu.linalg`` ->
  ``cupy.linalg``), so nothing is imported eagerly here beyond ``cupy``
  itself.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys

import cupy as _cupy

_ALIAS = __name__
_TARGET = 'cupy'


class _AliasLoader(importlib.abc.Loader):
    """Return the already-existing ``cupy.*`` module instead of a new one."""

    def __init__(self, target: str) -> None:
        self._target = target
        self._attrs: tuple = ()

    def create_module(self, spec):
        module = importlib.import_module(self._target)
        # `importlib._bootstrap._init_module_attrs` is about to stamp our
        # alias spec onto the real module; remember the originals so that
        # `exec_module` can put them back.
        self._attrs = (
            module.__name__, module.__spec__, module.__loader__,
        )
        return module

    def exec_module(self, module) -> None:
        # Restore, so `importlib.reload()` / introspection of the real
        # `cupy.*` module keep working.
        module.__name__, module.__spec__, module.__loader__ = self._attrs


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolve ``numpy_xpu`` and ``numpy_xpu.*`` to the ``cupy`` tree."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _ALIAS and not fullname.startswith(_ALIAS + '.'):
            return None
        real = _TARGET + fullname[len(_ALIAS):]
        if real not in sys.modules:
            try:
                if importlib.util.find_spec(real) is None:
                    return None
            except (ImportError, AttributeError, ValueError):
                return None
        # No `origin` (keeps the real `__file__`) and no search locations
        # (keeps the real `__path__`).
        return importlib.machinery.ModuleSpec(
            fullname, _AliasLoader(real))


def _install_alias() -> None:
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
    # Replace this module object by the `cupy` one: after the import system
    # finishes executing us it re-reads `sys.modules[__name__]`, so every
    # `import numpy_xpu` from now on yields `cupy` itself.
    sys.modules[_ALIAS] = _cupy


_install_alias()
