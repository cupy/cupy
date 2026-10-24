#!/usr/bin/env python
"""Copy the CUDA Toolkit fp16 headers into the runtime include tree.

Why this script exists
----------------------
``cuda_fp16.h`` / ``cuda_fp16.hpp`` are **NVIDIA proprietary** files and are
therefore no longer vendored in this repository (see
``docs/ascend/Compliance.md``).  The CUDA backend still needs them at run time
for NVRTC JIT on CUDA 11.x / 12.0-12.1 (CUDA Enhanced Compatibility), but they
are only ever *referenced*: they are copied from the user's own local toolkit
into a **gitignored** directory, and both ``setup.py`` and ``MANIFEST.in``
filter that directory out of every wheel/sdist.

This script never downloads anything from the network -- it only copies files
that already exist on this machine (CUDA Toolkit, conda ``cuda-*`` package,
``nvidia-cuda-runtime-*`` pip wheel).  Ascend/CANN builds do not need it.

Usage
-----
    python tools/fetch_cuda_fp16_headers.py --list
    python tools/fetch_cuda_fp16_headers.py                 # auto-detect
    python tools/fetch_cuda_fp16_headers.py --include-dir /usr/local/cuda/include
    python tools/fetch_cuda_fp16_headers.py --target cuda-12 --force

Exit codes: 0 = copied / already present, 1 = not found (or nothing to do).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

HEADERS = ('cuda_fp16.h', 'cuda_fp16.hpp')

#: ``_cuda/<dir>`` -> (min major, minor) that the bundled copy serves.
#: Mirrors cupy/_core/_gpu/_compile_with_cache.pyx: CUDA 12.2+ must not bundle.
TARGETS = {
    'cuda-11': (11, 0),
    'cuda-12': (12, 0),
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST_ROOT = os.path.join(
    REPO_ROOT, 'cupy', '_core', 'include', 'cupy', '_cuda')


def _version_from_header(include_dir):
    """Parse ``#define CUDA_VERSION`` from ``<include>/cuda.h``."""
    cuda_h = os.path.join(include_dir, 'cuda.h')
    try:
        with open(cuda_h, encoding='utf-8', errors='ignore') as f:
            m = re.search(r'#define\s+CUDA_VERSION\s+(\d+)', f.read())
    except OSError:
        return None
    if not m:
        return None
    v = int(m.group(1))
    return (v // 1000, (v % 1000) // 10)


def _version_from_version_json(include_dir):
    """Parse the toolkit's ``version.json`` (two levels above ``include``)."""
    path = os.path.join(os.path.dirname(include_dir), 'version.json')
    try:
        with open(path, encoding='utf-8') as f:
            version = json.load(f).get('version', '')
    except (OSError, ValueError):
        return None
    parts = version.split('.')
    if len(parts) < 2 or not parts[0].isdigit():
        return None
    return (int(parts[0]), int(parts[1]))


def _version_from_nvcc():
    try:
        out = subprocess.run(['nvcc', '--version'], capture_output=True,
                             text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r'release (\d+)\.(\d+)', out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _detect_version(include_dir):
    for fn in (_version_from_header, _version_from_version_json):
        v = fn(include_dir)
        if v:
            return v
    return None


def _site_package_includes():
    """``include`` dirs shipped by ``nvidia-cuda-runtime-*`` / conda wheels."""
    try:
        import site
        roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    except Exception:
        roots = []
    try:
        import sysconfig
        roots.append(sysconfig.get_paths().get('purelib', ''))
    except Exception:
        pass
    found = []
    for root in filter(None, roots):
        for sub in ('nvidia/cuda_runtime/include', 'nvidia/cuda_nvcc/include'):
            path = os.path.join(root, sub)
            if os.path.isfile(os.path.join(path, HEADERS[0])):
                found.append(path)
    return found


def candidate_include_dirs():
    """Ordered list of ``include`` dirs that may hold ``cuda_fp16.h``."""
    cands = []
    for env in ('CUDA_HOME', 'CUDA_PATH', 'CUDA_ROOT'):
        root = os.environ.get(env)
        if root:
            cands.append(os.path.join(root, 'include'))
    conda = os.environ.get('CONDA_PREFIX')
    if conda:
        cands.append(os.path.join(conda, 'include'))
    # /usr/local/cuda plus versioned siblings, newest first.
    cands.append('/usr/local/cuda/include')
    versioned = sorted(glob.glob('/usr/local/cuda-*/include'), reverse=True)
    cands.extend(versioned)
    cands.extend(_site_package_includes())
    seen, ordered = set(), []
    for c in cands:
        c = os.path.normpath(c)
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def pick_target(version):
    """Map a CUDA version to a bundled include dir name (or None)."""
    if version is None:
        return None
    major, minor = version
    if major == 11:
        return 'cuda-11'
    if major == 12 and minor < 2:
        return 'cuda-12'
    return None


def _list_candidates():
    found = False
    for include_dir in candidate_include_dirs():
        has = all(os.path.isfile(os.path.join(include_dir, h))
                  for h in HEADERS)
        if not has:
            continue
        found = True
        version = _detect_version(include_dir)
        target = pick_target(version)
        print('  {}{}  version={}  -> {}'.format(
            include_dir,
            '' if os.path.isdir(include_dir) else ' (missing)',
            '?.?' if version is None else '%d.%d' % version,
            target or 'not needed (no bundling for this version)'))
    if not found:
        print('  no local CUDA Toolkit / conda / nvidia-* wheel headers found')
    nvcc = _version_from_nvcc()
    if nvcc:
        print('  nvcc reports CUDA %d.%d' % nvcc)
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Copy CUDA Toolkit fp16 headers into '
                    'cupy/_core/include/cupy/_cuda/ (gitignored, local only).')
    parser.add_argument('--include-dir', action='append', default=[],
                        help='extra CUDA include dir to search (repeatable)')
    parser.add_argument('--target', choices=sorted(TARGETS),
                        help='force the destination dir instead of detecting')
    parser.add_argument('--list', action='store_true',
                        help='only list candidates and exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='show what would be copied')
    parser.add_argument('--force', action='store_true',
                        help='overwrite an existing copy')
    args = parser.parse_args(argv)

    print('CUDA fp16 headers are NOT vendored (docs/ascend/Compliance.md).')
    print('Candidates:')
    if args.include_dir:
        for d in args.include_dir:
            print('  (from --include-dir) {}'.format(os.path.normpath(d)))
    _list_candidates()
    if args.list:
        return 0

    search = list(args.include_dir) + candidate_include_dirs()
    for include_dir in search:
        include_dir = os.path.normpath(include_dir)
        if not all(os.path.isfile(os.path.join(include_dir, h))
                   for h in HEADERS):
            continue
        version = _detect_version(include_dir) or _version_from_nvcc()
        target = args.target or pick_target(version)
        if target is None:
            print('\n{}: no bundled include dir is needed for CUDA {} -- '
                  'nothing to do.'.format(
                      include_dir,
                      'unknown' if version is None
                      else '%d.%d' % version))
            return 1
        dest = os.path.join(DEST_ROOT, target)
        print('\nsource: {}\ndest:   {}'.format(include_dir, dest))
        if args.dry_run:
            for h in HEADERS:
                print('  would copy {}'.format(h))
            return 0
        os.makedirs(dest, exist_ok=True)
        for h in HEADERS:
            src = os.path.join(include_dir, h)
            dst = os.path.join(dest, h)
            if os.path.exists(dst) and not args.force:
                print('  skip (exists): {}'.format(dst))
                continue
            shutil.copyfile(src, dst)
            print('  copied {} -> {}'.format(src, dst))
        print('\nDone. This copy is gitignored and filtered out of '
              'sdist/wheel; do not commit or redistribute it.')
        return 0

    print('\nNot found. Provide the headers from your own CUDA toolkit, e.g.:\n'
          '  export CUDA_HOME=/usr/local/cuda && '
          'python tools/fetch_cuda_fp16_headers.py\n'
          'or install a runtime wheel and re-run:\n'
          '  pip install nvidia-cuda-runtime-cu12\n'
          'Ascend/CANN builds do not need these headers at all.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
