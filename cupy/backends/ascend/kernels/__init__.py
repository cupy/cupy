"""Built-in custom AscendC kernels for numpy-ascend.

Each entry maps a public ufunc (without the ``ascend_`` prefix) to an entry
point in an AscendC fatbin. Binaries are JIT-compiled on first import by
:func:`ensure_built` and cached on disk; see docs/ascend/CustomKernel.md.

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


def _build_dir() -> str:
    return os.path.join(os.path.dirname(__file__), '_build')


def ensure_built(force: bool = False) -> dict[str, str]:
    """JIT-compile all kernel sources (cached). Returns {bin name: fatbin path}.

    Raises RuntimeError when bisheng is unavailable or compilation fails;
    callers should treat that as "custom kernels disabled".
    """
    out_dir = _build_dir()
    os.makedirs(out_dir, exist_ok=True)
    soc = bisheng.default_soc()
    built: dict[str, str] = {}
    here = os.path.dirname(__file__)
    for stem, src_name in KERNEL_SOURCES.items():
        src = os.path.join(here, src_name)
        out = os.path.join(out_dir, f'{stem}_{soc}.o')
        stale = (force or not os.path.isfile(out)
                 or os.path.getmtime(src) > os.path.getmtime(out))
        if stale:
            bisheng.compile_kernel(src, out, soc=soc)
        built[stem] = out
    return built
