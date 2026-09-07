#!/usr/bin/env python3
"""
Prepare the CuPy source tree for building an sdist (``cupy-<version>.tar.gz``).

Writes ``description.rst`` (becomes the sdist's ``long_description``) and
prints env vars (one per line, ``KEY=VALUE``) to stdout for the caller to
append to ``$GITHUB_ENV`` or eval in a shell.

Local reproduction::

    python ci/tools/prepare_sdist_build.py >> .env
    set -a; . ./.env; set +a
    CUPY_INSTALL_USE_STUB=1 python -m build --sdist
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wheel_configs import SDIST_LONG_DESCRIPTION  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    description_path = REPO_ROOT / "description.rst"
    description_path.write_text(SDIST_LONG_DESCRIPTION, encoding="utf-8")
    print(f"CUPY_INSTALL_LONG_DESCRIPTION={description_path.resolve()}")
    print("CUPY_INSTALL_USE_STUB=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
