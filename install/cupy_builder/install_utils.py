from __future__ import annotations

import os


def print_warning(*lines: str) -> None:
    print('**************************************************')
    for line in lines:
        print('*** WARNING: %s' % line)
    print('**************************************************')


def get_path(key: str) -> list[str]:
    return os.environ.get(key, '').split(os.pathsep)


def search_on_path(filenames: list[str]) -> str | None:
    for p in get_path('PATH'):
        for filename in filenames:
            full = os.path.join(p, filename)
            if os.path.exists(full):
                return os.path.abspath(full)
    return None
