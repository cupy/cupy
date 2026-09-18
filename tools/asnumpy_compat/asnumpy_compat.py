"""Run CuPy's test suite against **asnumpy** (NumPy-on-Ascend) by
monkey-patching the ``cupy`` module to act as a facade over asnumpy.

Why this works
--------------
CuPy's tests never import another array library: they receive the *module
object itself* as an ``xp`` argument and call ``xp.<api>`` on it::

    @testing.numpy_cupy_allclose()
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)          # xp IS the cupy module

    # cupy/testing/_loops.py:72
    kw[name] = cupy
    # cupy/testing/_array.py:33 (device -> host conversion)
    numpy.testing.assert_allclose(cupy.asnumpy(actual), cupy.asnumpy(desired))

All of these are *runtime dynamic attribute lookups on the cupy module*, so
rebinding ``cupy.<attr>`` to asnumpy equivalents is enough: no cupy array is
ever created, and cupy's whole testing infrastructure (dtype parameterization,
numpy-vs-xp error compatibility, assertion helpers) is reused as-is.

Rule of safety: every ``cupy`` public name that asnumpy cannot provide is
shadowed with a stub that raises ``NotImplementedError``.  We deliberately do
NOT fall back to the real cupy implementation, otherwise a missing asnumpy
feature would silently be tested against the aclnn backend instead, making the
run meaningless.

Usage
-----
    pip install -e /home/qingfeng/repos/asnumpy      # once
    cd /home/qingfeng/numpy-ascend
    PYTHONPATH=tools/asnumpy_compat pytest tests/cupy_tests/math_tests \
        -p asnumpy_compat --ascend-dtype-filter=on -q

The plugin installs the patch in ``pytest_configure`` and restores the original
``cupy`` attributes in ``pytest_unconfigure``.  Whole test modules that depend
on cupy internals (cuda streams, cupyx, array_api, ...) are skipped during
collection; override with ``ASNUMPY_COMPAT_ALLOW`` (comma separated patterns).

Note: cupy's ascend dtype filter (float64/complex skipping) is keyed on the
compiled-in backend.  When running against asnumpy, decide explicitly:
``--ascend-dtype-filter=off`` if asnumpy supports float64/complex on the NPU,
``on``/``auto`` otherwise.
"""

import os
import sys
import types as _types

import pytest

# ---------------------------------------------------------------------------
# install / restore
# ---------------------------------------------------------------------------

_orig_attrs = None


def _host(x):
    """Device -> host conversion (replacement for ``cupy.asnumpy``)."""
    import numpy
    if hasattr(x, 'to_numpy'):          # asnumpy.ndarray
        return x.to_numpy()
    if hasattr(x, 'get'):               # cupy.ndarray (in case one leaks in)
        return x.get()
    if hasattr(x, '__dlpack__'):
        return numpy.from_dlpack(x)
    arr = numpy.asarray(x)
    if (arr.dtype == numpy.dtype(object) and not isinstance(x, numpy.ndarray)):
        raise TypeError(
            f'asnumpy-compat: cannot convert {type(x)!r} to a host array')
    return arr


class _Missing:
    """Callable attribute that fails loudly instead of falling back to cupy."""

    def __init__(self, qualname):
        self._qualname = qualname

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f'asnumpy-compat: cupy.{self._qualname} has no asnumpy mapping')

    def __getattr__(self, item):
        raise NotImplementedError(
            f'asnumpy-compat: cupy.{self._qualname}.{item} has no asnumpy '
            f'mapping')

    def __repr__(self):
        return f'<asnumpy-compat MISSING cupy.{self._qualname}>'


_SUBMODULE_MAP = {
    'linalg': 'asnumpy.linalg',
    'random': 'asnumpy.random',
}

_STUBBED_SUBMODULES = ('fft', 'polynomial')

_DENYLIST = [
    # depend on cupy's compiled core / backend, not on the cupy.* API surface
    'cupy_tests/cuda_tests',
    'cupy_tests/core_tests',
    'cupy_tests/import_tests',
    'cupy_tests/install_tests',
    'cupy_tests/typing_tests',
    'cupy_tests/test_init',
    'cupy_tests/test_typing',
    'cupy_tests/array_api_tests',
    # real cupyx (sparse / scipy) — not covered by the facade
    'cupyx_tests',
    'cupy_tests/testing_tests',   # tests cupy.testing itself via pytester
]


def _install():
    global _orig_attrs
    import cupy
    import cupyx  # noqa: F401  (must load before patching, kept real)
    import asnumpy
    import numpy

    if _orig_attrs is not None:
        return
    _orig_attrs = dict(cupy.__dict__)

    def backend_attr(path):
        obj = asnumpy
        for part in path.split('.'):
            obj = getattr(obj, part)
        return obj

    names = set(getattr(cupy, '__all__', ()))
    names |= {n for n in dir(cupy) if not n.startswith('_')}
    names |= {'ndarray', 'asnumpy', 'get_array_module', 'is_available'}

    for name in sorted(names):
        orig = _orig_attrs.get(name)
        # exceptions / warnings / dtype aliases stay real
        if isinstance(orig, type) and issubclass(orig, (BaseException, Warning)):
            continue
        if name.startswith('_') or name in ('testing', 'cupyx'):
            continue

        if name == 'asnumpy':
            setattr(cupy, name, _host)
            continue
        if name == 'get_array_module':
            setattr(cupy, name, lambda *a: cupy)
            continue
        if name == 'is_available':
            setattr(cupy, name, lambda: True)
            continue
        if name == 'ndarray':
            setattr(cupy, name, asnumpy.ndarray)
            continue
        if name in _SUBMODULE_MAP:
            try:
                setattr(cupy, name, backend_attr(_SUBMODULE_MAP[name]))
            except AttributeError:
                setattr(cupy, name, _Missing(name))
            continue
        if name in _STUBBED_SUBMODULES:
            setattr(cupy, name, _Missing(name))
            continue

        try:
            setattr(cupy, name, getattr(asnumpy, name))
        except AttributeError:
            if callable(orig) or isinstance(orig, type):
                # shadow: never let the real cupy backend answer
                setattr(cupy, name, _Missing(name))
            elif isinstance(orig, _types.ModuleType):
                # unmapped submodule (cuda, backends, core, ...): shadow so no
                # test can silently exercise the real cupy backend
                setattr(cupy, name, _Missing(name))
            # plain data attributes are left alone

    # submodule-level patches: cupy.linalg / cupy.random / cupy.fft
    for sub, backend_path in _SUBMODULE_MAP.items():
        mod = sys.modules.get(f'cupy.{sub}')
        if mod is None:
            continue
        try:
            backend = backend_attr(backend_path)
        except AttributeError:
            backend = None
        for n in {x for x in dir(mod) if not x.startswith('_')}:
            if backend is not None and hasattr(backend, n):
                setattr(mod, n, getattr(backend, n))
            elif callable(getattr(mod, n, None)) or isinstance(
                    getattr(mod, n, None), type):
                setattr(mod, n, _Missing(f'{sub}.{n}'))
    for sub in _STUBBED_SUBMODULES:
        mod = sys.modules.get(f'cupy.{sub}')
        if mod is None:
            continue
        for n in {x for x in dir(mod) if not x.startswith('_')}:
            if callable(getattr(mod, n, None)) or isinstance(
                    getattr(mod, n, None), type):
                setattr(mod, n, _Missing(f'{sub}.{n}'))

    # sanity: numpy scalars/dtypes must pass through _host untouched
    assert _host(numpy.float32(1)) == 1.0


def _restore():
    global _orig_attrs
    if _orig_attrs is None:
        return
    import cupy
    cupy.__dict__.clear()
    cupy.__dict__.update(_orig_attrs)
    _orig_attrs = None


# ---------------------------------------------------------------------------
# pytest plugin
# ---------------------------------------------------------------------------

def pytest_configure(config):
    try:
        _install()
        print('\nasnumpy-compat: cupy facade installed '
              '(backend: asnumpy, arrays are asnumpy.ndarray)')
    except ImportError as e:
        raise pytest.UsageError(
            'asnumpy-compat: cannot import asnumpy '
            '(pip install -e ~/repos/asnumpy first): %s' % e)


def pytest_unconfigure(config):
    _restore()


def pytest_collection_modifyitems(config, items):
    allow = os.environ.get('ASNUMPY_COMPAT_ALLOW', '')
    allow_pats = [p for p in allow.replace(',', ':').split(':') if p]
    deny_pats = _DENYLIST
    skipped = 0
    for item in items:
        path = str(item.fspath)
        if any(p in path for p in allow_pats):
            continue
        if any(p in path for p in deny_pats):
            item.add_marker(pytest.mark.skip(
                reason='asnumpy-compat: depends on cupy internals/cupyx'))
            skipped += 1
    if skipped:
        print(f'asnumpy-compat: {skipped} items skipped (denylist)')
