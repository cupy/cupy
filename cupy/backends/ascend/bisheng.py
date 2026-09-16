"""bisheng (AscendC device compiler) wrapper for numpy-ascend custom kernels.

Compiles AscendC kernel sources into aicore fatbins (.o) that are loaded at
runtime through the aclrt binary API (see acl_custom_kernels.h). This is the
"JIT" half of the design in docs/ascend/CustomKernel.md -- the wheel ships
sources only; any machine with a CANN install has bisheng.
"""

import os
import platform
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


def _arch_roots(cann_root: str) -> list[str]:
    """CANN's machine-specific sub-directories (``aarch64-linux``/``x86_64-linux``).

    The AscendC headers live under the **host** arch dir: NPU boxes are usually
    aarch64 while dev boxes are x86_64. Hard-coding ``x86_64-linux`` (what this
    used to do) means ``kernel_operator.h`` is not found on an aarch64 machine
    even though it is right there under ``aarch64-linux``.
    """
    machine = platform.machine().lower()
    if machine in ('aarch64', 'arm64'):
        preferred = 'aarch64-linux'
    elif machine in ('x86_64', 'amd64'):
        preferred = 'x86_64-linux'
    else:
        preferred = ''
    roots = []
    for name in dict.fromkeys((preferred, 'aarch64-linux', 'x86_64-linux')):
        if not name:
            continue
        path = os.path.join(cann_root, name)
        if os.path.isdir(path):
            roots.append(path)
    return roots


def _ascendc_include_dirs(cann_root: str) -> list[str]:
    """AscendC header roots (layout differs between CANN releases; keep what exists)."""
    dirs = []
    # Bail out to the toolkit root itself when there is no per-arch dir, so that
    # an unpacked AscendC SDK (<root>/include/...) still contributes -I paths.
    for root in _arch_roots(cann_root) or [cann_root]:
        # CANN 9.0 full SDK: framework under <arch>/asc, AscendC headers under
        # <arch>/asc/include (that is where `kernel_operator.h` lives).
        asc = os.path.join(root, 'asc')
        if os.path.isdir(asc):
            dirs += [
                os.path.join(asc, 'include'),
                os.path.join(asc, 'include', 'basic_api'),
                os.path.join(asc, 'impl', 'basic_api'),
            ]
        # standalone AscendC SDK: the tarball ships an extra `include` level,
        # i.e. <arch>/ascendc/include/include/basic_api
        for ascendc in (
                os.path.join(root, 'ascendc', 'include', 'include', 'basic_api'),
                os.path.join(root, 'ascendc', 'include', 'include'),
                os.path.join(root, 'ascendc', 'include', 'basic_api'),
        ):
            if os.path.isdir(ascendc):
                dirs.append(ascendc)
        # highlevel math lib (Trunc/Frac/Floor/Log2/Atan/...)
        for hl in (os.path.join(root, 'ascendc', 'include', 'highlevel_api'),
                   os.path.join(asc, 'include', 'highlevel_api')):
            if os.path.isdir(hl):
                dirs.append(hl)
        # CANN 8.5 layout fallback (tikcfw) -- also the only place that carries
        # `lib/math/*.h` on 9.0, so keep it after the asc dirs.
        tik = os.path.join(root, 'tikcpp', 'tikcfw')
        if os.path.isdir(tik):
            dirs += [tik, os.path.join(tik, 'interface'), os.path.join(tik, 'impl')]
    # de-duplicate (host arch + fallback arch dirs may both exist), drop missing
    return [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]


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
