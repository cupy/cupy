"""Built-in custom AscendC kernels for numpy-ascend.

Each entry maps a public ufunc (without the ``ascend_`` prefix) to an entry
point in an AscendC fatbin. See docs/ascend/CustomKernel.md.

Fatbin resolution order (see :func:`ensure_built`):

1. ``_aot/<source stem>_<soc>.o`` -- prebuilt by ``setup.py build_ext`` and
   shipped inside the wheel (``CUPY_ASCEND_KERNEL_SOCS``). Nothing is compiled
   at import time.
2. ``_build/<source stem>_<soc>.o`` -- JIT cache, written on first import for a
   SoC the wheel was not built for (source build, or another Ascend model).

The SoC in the file name is what tells the hardware apart: one directory can
hold fatbins for several Ascend models. See ``bisheng.compiled_path``.

Verification level: L1 (compiles). Numerical correctness requires a 910B.
"""

import os

from cupy.backends.ascend import bisheng

#: source stem -> source file name
KERNEL_SOURCES = {
    'elementwise': 'ascendc_elementwise.cpp',
}

#: public ufunc name -> kernel spec
CUSTOM_UFUNCS = {
    # B-tier elementwise APIs with no aclnn equivalent (plan.md §2 B档)
    'conjugate': {
        'bin': 'elementwise', 'entry': 'ascendc_conj_f32',
        'n_out': 1, 'n_in': 1, 'dtypes': ('F',),
    },
    'angle': {
        'bin': 'elementwise', 'entry': 'ascendc_angle_f32',
        'n_out': 1, 'n_in': 1, 'dtypes': ('F',),
    },
    'imag': {
        'bin': 'elementwise', 'entry': 'ascendc_imag_f32',
        'n_out': 1, 'n_in': 1, 'dtypes': ('F',),
    },
    'frexp': {
        'bin': 'elementwise', 'entry': 'ascendc_frexp_f32',
        'n_out': 2, 'n_in': 1, 'dtypes': ('f',),
    },
    'modf': {
        'bin': 'elementwise', 'entry': 'ascendc_modf_f32',
        'n_out': 2, 'n_in': 1, 'dtypes': ('f',),
    },
    'ldexp': {
        'bin': 'elementwise', 'entry': 'ascendc_ldexp_f32',
        'n_out': 1, 'n_in': 2, 'dtypes': ('f',),
    },
    # NOTE: left_shift / right_shift used to live here, but they are covered
    # by the builtin aclnn registrations in acl_utils.pyx (aclop_LeftShift /
    # aclop_RightShift), so the custom kernels were removed — a custom kernel
    # registered here would silently overwrite the working builtin.
}


#: Directory (next to this file) with the fatbins the wheel ships, written by
#: ``setup.py build_ext`` for the SoCs in ``CUPY_ASCEND_KERNEL_SOCS``.
#: Layout: ``_aot/<source stem>_<soc>.o``. Read-only at runtime.
AOT_DIR_NAME = '_aot'

#: Runtime JIT cache, same layout. Written on first import for a SoC that
#: ``_aot`` does not cover.
JIT_DIR_NAME = '_build'


def aot_dir() -> str:
    """Directory of the fatbins shipped inside the wheel (read-only)."""
    return os.path.join(os.path.dirname(__file__), AOT_DIR_NAME)


def jit_dir() -> str:
    """Directory of the fatbins compiled on this machine (JIT cache)."""
    return os.path.join(os.path.dirname(__file__), JIT_DIR_NAME)


def aot_socs() -> list[str]:
    """SoCs this install ships prebuilt fatbins for.

    Recovered from the artifact names (``<source stem>_<soc>.o``). Empty on a
    source build without the AOT step, or when the wheel was built for other
    Ascend models. Diagnostic helper -- :func:`ensure_built` silently falls back
    to JIT for every SoC not listed here.
    """
    root = aot_dir()
    try:
        names = os.listdir(root)
    except OSError:
        return []
    # Longest stem first: one stem may be a prefix of another.
    stems = sorted((os.path.splitext(name)[0] for name in KERNEL_SOURCES.values()),
                   key=len, reverse=True)
    socs = set()
    for name in names:
        if not name.endswith('.o'):
            continue
        base = name[:-len('.o')]
        for stem in stems:
            if base.startswith(stem + '_'):
                socs.add(base[len(stem) + 1:])
                break
    return sorted(socs)


def _built_path(src_name: str, soc: str, force: bool = False) -> str:
    """Fatbin for one source: the wheel's copy if present, else JIT into cache."""
    stem = os.path.splitext(src_name)[0]
    aot_bin = bisheng.compiled_path(aot_dir(), soc, stem)
    if not force and os.path.isfile(aot_bin):
        return aot_bin
    src = os.path.join(os.path.dirname(__file__), src_name)
    out = bisheng.compiled_path(jit_dir(), soc, stem)
    if force or bisheng.is_stale(src, out):
        bisheng.compile_kernel(src, out, soc=soc)
    return out


def ensure_built(force: bool = False) -> dict[str, str]:
    """Return {registry bin name: fatbin path}, compiling only when needed.

    A fatbin already built for the running SoC (``_aot/<stem>_<soc>.o``,
    produced by ``setup.py build_ext`` and shipped inside the wheel) is used
    as-is -- bisheng is never invoked, which is the whole point of the AOT step.
    Only an uncovered SoC (or ``force``) falls back to JIT-compiling into
    ``_build/<stem>_<soc>.o``.

    The ``_aot`` lookup is by existence only, never by mtime: inside an
    installed wheel both files carry the extraction time, so an mtime
    comparison would be arbitrary (and falling back to JIT in a read-only
    ``site-packages`` can not even write its cache). Consequence for a *source*
    checkout: ``build_ext`` refreshes ``_aot``, but editing ``kernels/*.cpp``
    without rebuilding leaves a stale fatbin in place -- re-run
    ``build_ext --inplace`` (or remove ``_aot``) after editing a kernel.

    Raises RuntimeError when bisheng is unavailable or compilation fails;
    callers should treat that as "custom kernels disabled".
    """
    soc = bisheng.default_soc()
    return {
        stem: _built_path(src_name, soc, force)
        for stem, src_name in KERNEL_SOURCES.items()
    }
