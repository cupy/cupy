#!/usr/bin/env python3
"""
Prepare a CuPy source tree for building a release-ready wheel.

This script:

* Rewrites ``pyproject.toml``'s ``project.name`` to the CTK-major-specific
  package (``cupy-cuda12x`` / ``cupy-cuda13x``).
* Writes ``description.rst`` (becomes the wheel's ``long_description``).
* Generates ``_wheel.json`` preload metadata for the ``[ctk]`` runtime extras.
* Globs ``--preload-dir`` for include/lib subdirs to point CuPy at.
* Prints env vars (one per line, ``KEY=VALUE``) to stdout for the caller to
  append to ``$GITHUB_ENV`` or eval in a shell.

The caller is responsible for first running
``cupyx/tools/install_library.py`` to populate ``--preload-dir``.

Local reproduction::

    python cupyx/tools/install_library.py --library cutensor --cuda 13.x --arch x86_64 --prefix ./preloads --action install
    python cupyx/tools/install_library.py --library nccl     --cuda 13.x --arch x86_64 --prefix ./preloads --action install
    python ci/tools/prepare_wheel_build.py --cuda-major 13 --host-platform linux >> .env
    set -a; . ./.env; set +a
    pip wheel .
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]
import tomli_w

# Make sibling wheel_configs.py importable when invoked from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wheel_configs import (  # noqa: E402
    PRELOAD_LIBRARIES,
    WHEEL_LONG_DESCRIPTION_CUDA,
    WHEEL_PACKAGE_NAMES,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def rename_project(cuda_major: str) -> str:
    """Rewrite ``project.name`` in ``pyproject.toml`` in place."""
    pyproject = REPO_ROOT / "pyproject.toml"
    package_name = WHEEL_PACKAGE_NAMES[cuda_major]
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    data["project"]["name"] = package_name
    with pyproject.open("wb") as f:
        tomli_w.dump(data, f)
    return package_name


def write_long_description(cuda_major: str) -> Path:
    description_path = REPO_ROOT / "description.rst"
    description = WHEEL_LONG_DESCRIPTION_CUDA.format(
        version=f"{cuda_major}.x",
        wheel_suffix=f"{cuda_major}x",
    )
    description_path.write_text(description, encoding="utf-8")
    return description_path


def generate_wheel_metadata(cuda_major: str, host_platform: str) -> Path:
    target_system = {
        "linux": "Linux:x86_64",
        "win": "Windows:x86_64",
    }[host_platform]

    libraries = PRELOAD_LIBRARIES[cuda_major][host_platform]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "cupyx" / "tools" / "_generate_wheel_metadata.py"),
        "--cuda", f"{cuda_major}.x",
        "--target", target_system,
    ]
    for library in libraries:
        cmd.extend(["--library", library])

    output = subprocess.check_output(cmd, encoding="utf-8")
    json.loads(output)  # Sanity-check JSON; raises if invalid.
    metadata_path = REPO_ROOT / "_wheel.json"
    metadata_path.write_text(output, encoding="utf-8")
    return metadata_path


def collect_preload_paths(
    preload_dir: Path, cuda_major: str, host_platform: str
) -> tuple[list[Path], list[Path]]:
    """Discover ``include`` and ``lib`` subdirs under ``preload_dir``.

    ``cupyx/tools/install_library.py`` installs to
    ``{prefix}/{cuda}/{lib}/{lib_version}/...`` (see ``calculate_destination``).
    NCCL extracts to a vendor layout (``include/``, ``lib/``). cuTENSOR
    installs ``include/`` and ``lib/`` siblings.
    """
    libraries = PRELOAD_LIBRARIES[cuda_major][host_platform]
    include_dirs: list[Path] = []
    library_dirs: list[Path] = []
    cuda_subdir = preload_dir / f"{cuda_major}.x"
    for library in libraries:
        lib_root = cuda_subdir / library
        if not lib_root.exists():
            continue
        for version_dir in sorted(lib_root.iterdir()):
            if (version_dir / "include").is_dir():
                include_dirs.append(version_dir / "include")
            for lib_subdir in ("lib", "lib64"):
                if (version_dir / lib_subdir).is_dir():
                    library_dirs.append(version_dir / lib_subdir)
    return include_dirs, library_dirs


def _apply_prefix(path: Path, root_prefix: str | None) -> str:
    """Re-anchor a path under ``root_prefix`` (e.g. ``/project``).

    When ``root_prefix`` is ``None``, the host's absolute path is returned.
    Otherwise the path is made relative to ``REPO_ROOT`` and joined onto
    ``root_prefix`` so it works inside a cibuildwheel container.
    """
    if root_prefix is None:
        return str(path.resolve())
    rel = path.resolve().relative_to(REPO_ROOT)
    return os.path.join(root_prefix, str(rel))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-major", required=True, choices=sorted(WHEEL_PACKAGE_NAMES),
    )
    parser.add_argument(
        "--host-platform", required=True, choices=("linux", "win"),
        help="Build host: linux = Linux x86_64, win = Windows x86_64.",
    )
    parser.add_argument(
        "--preload-dir", type=Path, default=Path("./preloads"),
        help="Where cupyx/tools/install_library.py placed the preloads.",
    )
    parser.add_argument(
        "--root-prefix", default=None,
        help=(
            "Prefix to use when emitting paths (e.g. '/project' for "
            "cibuildwheel containers, or '/host'+abs-path-on-runner). "
            "Default: emit absolute host paths."
        ),
    )
    args = parser.parse_args(argv)

    package_name = rename_project(args.cuda_major)
    description_path = write_long_description(args.cuda_major)
    metadata_path = generate_wheel_metadata(args.cuda_major, args.host_platform)
    include_dirs, library_dirs = collect_preload_paths(
        args.preload_dir, args.cuda_major, args.host_platform,
    )

    sep = ";" if args.host_platform == "win" else ":"
    env_lines = [
        "CUPY_INSTALL_NO_RPATH=1",
        f"CUPY_INSTALL_LONG_DESCRIPTION={_apply_prefix(description_path, args.root_prefix)}",
        f"CUPY_INSTALL_WHEEL_METADATA={_apply_prefix(metadata_path, args.root_prefix)}",
        f"CUPY_PACKAGE_NAME={package_name}",
    ]
    if include_dirs:
        env_lines.append(
            "CUPY_INCLUDE_PATH=" + sep.join(
                _apply_prefix(p, args.root_prefix) for p in include_dirs
            )
        )
    if library_dirs:
        env_lines.append(
            "CUPY_LIBRARY_PATH=" + sep.join(
                _apply_prefix(p, args.root_prefix) for p in library_dirs
            )
        )

    print("\n".join(env_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
