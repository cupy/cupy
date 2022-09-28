import os
from typing import List, Optional


def print_warning(*lines: str) -> None:
    print('**************************************************')
    for line in lines:
        print('*** WARNING: %s' % line)
    print('**************************************************')


def get_path(key: str) -> List[str]:
    return os.environ.get(key, '').split(os.pathsep)


def search_on_path(filenames: List[str]) -> Optional[str]:
    for p in get_path('PATH'):
        for filename in filenames:
            full = os.path.join(p, filename)
            if os.path.exists(full):
                return os.path.abspath(full)
    return None
