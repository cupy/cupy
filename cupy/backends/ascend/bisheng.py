"""bisheng (AscendC device compiler) wrapper for numpy-ascend custom kernels.

Compiles AscendC kernel sources into aicore fatbins (.o) that are loaded at
runtime through the aclrt binary API (see acl_custom_kernels.h).

Two halves of the design in docs/ascend/CustomKernel.md:

* :func:`prebuild` -- AOT: ``setup.py build_ext`` compiles the built-in kernels
  for the SoCs named in ``CUPY_ASCEND_KERNEL_SOCS`` and the wheel ships the
  fatbins (``cupy/backends/ascend/kernels/_aot/<soc>/``).
* :func:`compile_kernel` -- JIT fallback for a SoC the wheel was not built for;
  any machine with a CANN install has bisheng.

Both write the same layout (:func:`compiled_path`), so the runtime lookup is a
plain ``os.path.isfile``.

This module deliberately imports only the standard library: the build system
loads it by path (see ``cupy_builder.backends.ascend``), because importing
``cupy`` from ``setup.py`` is not possible.
"""

import hashlib
import os
import platform
import subprocess
import sys
import tempfile
from typing import Iterable, Sequence


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
                   log_stream=None,
                   extra_options: Sequence[str] = ()) -> str:
    """Compile one AscendC source into an aicore fatbin. Returns ``out``."""
    root = _cann_root()
    bisheng = find_bisheng(root)
    if bisheng is None:
        raise RuntimeError(
            'bisheng compiler not found under CANN root; '
            'cannot build custom AscendC kernels')
    soc = soc or default_soc()
    # bisheng's internal `ar` step does not create the output directory, and it
    # reports the failure as "ar: ...: No such file or directory".
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cmd = [bisheng, '-O2', '-std=c++17', '-xcce']
    # An explicit --cce-soc-version in the user options takes priority over
    # the `soc` argument (later flags would win in clang anyway; skip ours to
    # avoid duplicate/contradictory flags).
    if not any(o.startswith('--cce-soc-version') for o in extra_options):
        cmd.append(f'--cce-soc-version={soc}')
    cmd += ['--cce-soc-core-type=VecCore', '--cce-build-static-lib']
    # User options come after the defaults so they can override them (clang
    # semantics: the last flag wins).
    cmd += list(extra_options)
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


# ---------------------------------------------------------------------------
# Artifact layout / batch AOT build
# ---------------------------------------------------------------------------
# The aicore ISA is SoC-specific, so a fatbin only runs on the SoC it was
# compiled for; that is why the SoC is part of the file name. Artifacts are
# named after the *source file* stem (not the registry key of
# `cupy/backends/ascend/kernels/__init__.py`) so that the build system can glob
# `kernels/*.cpp` without importing the `cupy` package.
def compiled_path(dest_dir: str, soc: str, stem: str) -> str:
    """Return ``<dest_dir>/<stem>_<soc>.o``, the fatbin path for one source.

    The SoC suffix is what tells the hardware apart: a wheel can carry fatbins
    for several Ascend models side by side, and ``unzip -l`` / a plain ``ls`` of
    the directory is self-describing (``ascendc_elementwise_Ascend910B4.o``).
    """
    return os.path.join(dest_dir, f'{stem}_{soc}.o')


def is_stale(src: str, out: str) -> bool:
    """Whether ``out`` is missing or older than its source ``src``."""
    return (not os.path.isfile(out)
            or os.path.getmtime(src) > os.path.getmtime(out))


def prebuild(srcs: Iterable[str], dest_dir: str, socs: Sequence[str],
             force: bool = False, log_stream=None) -> dict[str, list[str]]:
    """Compile every source in ``srcs`` for every SoC in ``socs``.

    Writes ``<dest_dir>/<source stem>_<soc>.o`` (:func:`compiled_path`) and skips
    artifacts that are already newer than their source, unless ``force``. Can
    populate a fresh ``_aot`` directory during a build, i.e. it does not need a
    pre-existing destination.

    Raises ``RuntimeError`` (via :func:`compile_kernel`) when bisheng is missing
    or a compilation fails. Returns ``{soc: [artifact paths]}``.
    """
    built: dict[str, list[str]] = {}
    for src in srcs:
        stem = os.path.splitext(os.path.basename(src))[0]
        for soc in socs:
            out = compiled_path(dest_dir, soc, stem)
            if force or is_stale(src, out):
                compile_kernel(src, out, soc=soc, log_stream=log_stream)
            built.setdefault(soc, []).append(out)
    return built


# ---------------------------------------------------------------------------
# JIT compile-with-cache: the NVRTC counterpart for AscendC kernels
#
# Mirrors the skeleton of ``cupy/cuda/compiler.py``'s
# ``_compile_module_with_cache`` (hash-keyed disk cache, atomic writes,
# error dump, source-integrity check) while keeping this module importable
# by the build system (standard library only, see the module docstring).
# The compiled payload is the aicore fatbin; loading it into a runnable
# module is the caller's job (``cupy.xpu.function.Module.load``).
# ---------------------------------------------------------------------------
#: Bumped whenever the on-disk cache layout below changes.
_kernel_cache_format_version = 1

#: SHA-1 hex digest length (same convention as cupy.cuda._compiler_cache).
_hash_length = 40


class BishengCompileException(Exception):
    """bisheng compilation error, mirroring ``cupy.cuda.compiler``'s
    ``CompileException`` (msg + source + options + dump())."""

    def __init__(self, msg, source, name, options, backend='bisheng'):
        self._msg = msg
        self.source = source
        self.name = name
        self.options = options
        self.backend = backend
        super().__init__()

    def __reduce__(self):
        return (type(self), (self._msg, self.source, self.name,
                             self.options, self.backend))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self._msg

    def dump(self, f):
        import math
        lines = self.source.split('\n')
        digits = int(math.floor(math.log10(len(lines)))) + 1
        linum_fmt = '{{:0{}d}} '.format(digits)
        f.write('{} '.format(self.backend.upper()))
        f.write('compilation error: {}\n'.format(self))
        f.write('-----\n')
        f.write('Name: {}\n'.format(self.name))
        f.write('Options: {}\n'.format(' '.join(self.options)))
        f.write('AscendC source:\n')
        for i, line in enumerate(lines):
            f.write(linum_fmt.format(i + 1) + line.rstrip() + '\n')
        f.write('-----\n')
        f.flush()


def kernel_cache_dir() -> str:
    """Directory for bisheng JIT artifacts.

    ``CUPY_CACHE_DIR`` overrides the base directory (same variable the CUDA
    side uses); otherwise ``~/.cache/cupy/cann_kernel_cache`` per
    docs/ascend/CustomKernel.md (decision #5).
    """
    base = os.environ.get('CUPY_CACHE_DIR',
                           os.path.expanduser('~/.cache/cupy'))
    return os.path.join(base, 'cann_kernel_cache')


_compiler_fingerprint_cache: str | None = None


def compiler_fingerprint() -> str:
    """A string identifying the exact compiler binary in use.

    Part of the cache key so a CANN upgrade (new bisheng) invalidates
    previously compiled fatbins. Cached in-process.
    """
    global _compiler_fingerprint_cache
    if _compiler_fingerprint_cache is None:
        root = _cann_root()
        path = find_bisheng(root)
        if path is None:
            _compiler_fingerprint_cache = 'no-bisheng'
        else:
            try:
                stat = os.stat(path)
                mtime = int(stat.st_mtime)
            except OSError:
                mtime = -1
            version = ''
            try:
                proc = subprocess.run([path, '--version'],
                                      capture_output=True, text=True)
                version = (proc.stdout or proc.stderr).strip().splitlines()
                version = version[0] if version else ''
            except OSError:
                pass
            _compiler_fingerprint_cache = f'{path}|{mtime}|{version}'
    return _compiler_fingerprint_cache


_cann_series_cache: str | None = None


def cann_series() -> str:
    """The CANN release train (``major.minor``) of the local install.

    Used in the cache key: per docs/ascend/CustomKernel.md, the key uses the
    *series* (not the patch level) so a 9.0.1-built cache stays valid on
    9.0.2, and cross-series upgrades force a recompile.
    """
    global _cann_series_cache
    if _cann_series_cache is None:
        root = _cann_root()
        series = 'unknown'
        if root:
            for rel in ('compiler/version.info', 'version.cfg',
                        'opp/version.info'):
                path = os.path.join(root, rel)
                try:
                    with open(path) as f:
                        content = f.read()
                except OSError:
                    continue
                import re
                m = re.search(r'(\d+\.\d+)\.', content) or \
                    re.search(r'(\d+\.\d+)(?!\.)', content)
                if m:
                    series = m.group(1)
                    break
        _cann_series_cache = series
    return _cann_series_cache


def _hash_hexdigest(value: bytes) -> str:
    return hashlib.sha1(value, usedforsecurity=False).hexdigest()


def _cache_key(source: str, options: Sequence[str], soc: str) -> str:
    """Cache file name for one (source, options, SoC) compilation.

    The key covers everything that can change the fatbin bytes: source,
    options, SoC, compiler binary (path+mtime+version), CANN series and the
    on-disk format version.
    """
    env = (soc, tuple(options), compiler_fingerprint(), cann_series(),
           'bisheng', _kernel_cache_format_version)
    key_src = '%s %s' % (env, source)
    return _hash_hexdigest(key_src.encode('utf-8')) + '.o'


def _cache_load(name: str) -> bytes | None:
    """Load a cached fatbin, verifying the SHA-1 payload prefix.

    On-disk layout (same as ``cupy.cuda._compiler_cache``): 40 ASCII hex
    chars of ``sha1(payload)`` followed by the payload. A truncated or
    corrupted entry is a miss, not an error.
    """
    path = os.path.join(kernel_cache_dir(), name)
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except OSError:
        return None
    if len(data) < _hash_length:
        return None
    payload = data[_hash_length:]
    if data[:_hash_length] != _hash_hexdigest(payload).encode('ascii'):
        return None
    return payload


def _cache_save(name: str, payload: bytes, source: str) -> None:
    """Atomically write a fatbin into the cache directory.

    Concurrent processes compiling the same key each write a temp file and
    ``os.replace`` it; last writer wins and readers only ever see complete
    files. Failure to write the cache is *not* an error (read-only HOME,
    full disk...): the caller already holds the payload in memory.
    """
    cache_dir = kernel_cache_dir()
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as tf:
            tf.write(_hash_hexdigest(payload).encode('ascii'))
            tf.write(payload)
            temp_path = tf.name
        try:
            os.replace(temp_path, os.path.join(cache_dir, name))
        except PermissionError:
            pass
        if os.environ.get('CUPY_CACHE_SAVE_CUDA_SOURCE'):
            with open(os.path.join(cache_dir, name + '.cce'), 'w') as f:
                f.write(source)
    except OSError:
        pass


def compile_module_with_cache(source: str, options: Sequence[str] = (),
                              *, soc: str | None = None,
                              log_stream=None) -> bytes:
    """Compile an AscendC source string into a fatbin, with disk cache.

    This is the bisheng counterpart of ``cupy.cuda.compiler
    ._compile_module_with_cache`` (CUDA: NVRTC in-process; Ascend: spawn the
    bisheng compiler, hence the mandatory disk cache). Returns the fatbin
    bytes, ready for ``cupy.xpu.function.Module.load``.

    Args:
        source: AscendC source code (``extern "C" __global__ __aicore__``
            kernels; ``kernel_operator.h`` include dirs are added
            automatically).
        options: extra bisheng command line options (e.g.
            ``('--cce-aicore-...',)``); passed through to the compiler and
            part of the cache key.
        soc: target SoC (default: :func:`default_soc`). Only used when the
            user options do not already carry a ``--cce-soc-version``.
        log_stream: writable object receiving the compiler log.
    """
    soc = soc or default_soc()
    name = _cache_key(source, options, soc)

    payload = _cache_load(name)
    if payload is not None:
        return payload

    with tempfile.TemporaryDirectory() as root_dir:
        src_path = os.path.join(root_dir, 'kern.cce')
        out_path = os.path.join(root_dir, name)
        with open(src_path, 'w') as f:
            f.write(source)
        try:
            compile_kernel(src_path, out_path, soc=soc,
                           log_stream=log_stream, extra_options=options)
        except RuntimeError as e:
            cex = BishengCompileException(str(e), source, name,
                                           tuple(options))
            if os.environ.get('CUPY_DUMP_CUDA_SOURCE_ON_ERROR'):
                cex.dump(sys.stderr)
            raise cex from e
        with open(out_path, 'rb') as f:
            payload = f.read()

    _cache_save(name, payload, source)
    return payload
