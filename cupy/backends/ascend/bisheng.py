"""bisheng (AscendC device compiler) wrapper for numpy-ascend custom kernels.

Compiles AscendC kernel sources into aicore fatbins (.o) that are loaded at
runtime through the aclrt binary API (see acl_custom_kernels.h). This is the
"JIT" half of the design in docs/ascend/CustomKernel.md -- the wheel ships
sources only; any machine with a CANN install has bisheng.
"""

import os
import subprocess


def _cann_root() -> str | None:
    root = os.environ.get('ASCEND_HOME_PATH')
    if root and os.path.isdir(root):
        return root
    root = os.environ.get('ASCEND_TOOLKIT_HOME')
    if root and os.path.isdir(root):
        return root
    for candidate in (
            '/usr/local/Ascend/ascend-toolkit/latest',
            os.path.expanduser('~/Ascend/cann'),):
        if os.path.isdir(candidate):
            return candidate
    return None


def find_bisheng(cann_root: str | None = None) -> str | None:
    """Locate the bisheng device compiler (moved between CANN releases)."""
    root = cann_root or _cann_root()
    if not root:
        return None
    # CANN <= 8.5: compiler/ccec_compiler/bin; CANN >= 9.0: tools/bisheng_compiler/bin
    candidates = [
        os.path.join(root, 'tools', 'bisheng_compiler', 'bin', 'bisheng'),
        os.path.join(root, 'compiler', 'ccec_compiler', 'bin', 'bisheng'),
        os.path.join(root, 'x86_64-linux', 'tools', 'bisheng_compiler', 'bin', 'bisheng'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _ascendc_include_dirs(cann_root: str) -> list[str]:
    """AscendC header roots (layout differs between CANN releases; keep what exists)."""
    dirs = []
    x86 = os.path.join(cann_root, 'x86_64-linux')
    # CANN 9.0: full framework under x86_64-linux/asc
    asc = os.path.join(x86, 'asc')
    if os.path.isdir(asc):
        dirs += [
            os.path.join(asc, 'include'),
            os.path.join(asc, 'include', 'basic_api'),
            os.path.join(asc, 'impl', 'basic_api'),
        ]
    # highlevel math lib (Trunc/Frac/Floor/Log2/Atan/...)
    for hl in (os.path.join(x86, 'ascendc', 'include', 'highlevel_api'),
               os.path.join(asc, 'include', 'highlevel_api')):
        if os.path.isdir(hl):
            dirs.append(hl)
    # CANN 8.5 layout fallback (tikcfw)
    for tik in (os.path.join(x86, 'tikcpp', 'tikcfw'),):
        if os.path.isdir(tik):
            dirs += [tik, os.path.join(tik, 'interface'), os.path.join(tik, 'impl')]
    return [d for d in dirs if os.path.isdir(d)]


def default_soc() -> str:
    """Target SoC. TODO: probe from the driver (npu-smi) instead of an env var."""
    return os.environ.get('CUPY_ASCEND_SOC', 'Ascend910B4')


def compile_kernel(src: str, out: str, soc: str | None = None,
                   log_stream=None) -> str:
    """Compile one AscendC source into an aicore fatbin. Returns ``out``."""
    root = _cann_root()
    bisheng = find_bisheng(root)
    if bisheng is None:
        raise RuntimeError(
            'bisheng compiler not found under CANN root; '
            'cannot build custom AscendC kernels')
    soc = soc or default_soc()
    cmd = [
        bisheng, '-O2', '-std=c++17', '-xcce',
        f'--cce-soc-version={soc}',
        '--cce-soc-core-type=VecCore',
        '--cce-build-static-lib',
    ]
    for d in _ascendc_include_dirs(root):
        cmd += ['-I', d]
    cmd += [src, '-o', out]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f'bisheng failed for {src} (exit {proc.returncode}):\n'
            f'cmd: {" ".join(cmd)}\n{proc.stderr[-4000:]}')
    if log_stream is not None and proc.stderr:
        log_stream.write(proc.stderr)
    return out
