"""Checks that xsf cannot be transitively included through any other headers.

Allows invalidation of kernel cache only for kernels directly including
something from xsf when xsf is updated, preventing expensive recompilation
of all kernels.

"""
#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import re
import sys

INCLUDE_DIR = Path(__file__).resolve().parents[1] / 'cupy/_core/include'
XSF_DIR = (INCLUDE_DIR / 'cupy/xsf').resolve()

INCLUDE_RE = re.compile(
    r'^[ \t]*#[ \t]*include[ \t]*[<"]([^>"]+)[>"]'
)


def includes_xsf(path, target):
    return any(
        (root / target).resolve().is_relative_to(XSF_DIR)
        for root in (path.parent, INCLUDE_DIR)
    )


def main():
    errors = []

    for path in INCLUDE_DIR.rglob('*'):
        if (
                not path.is_file()
                or path.suffix not in {".h", ".hpp", ".cuh"}
                or path.is_relative_to(XSF_DIR)
        ):
            continue

        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            match = INCLUDE_RE.match(line)
            if match and includes_xsf(path, match.group(1)):
                errors.append(
                    f'{path}:{lineno}: '
                    'xsf header included within non-xsf header.'
                )

    if errors:
        raise sys.exit('\n'.join(errors))


if __name__ == '__main__':
    main()
