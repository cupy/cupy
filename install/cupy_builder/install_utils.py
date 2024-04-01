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

def get_rocm_version() -> int:
    rocm_version = -1
    if os.getenv("ROCM_HOME"):
        rocm_home = str(os.getenv("ROCM_HOME"))
        version_path = os.path.join(rocm_home, ".info", "version")
        rocm_version = int(
            open(version_path).read().split("-")[0].replace(".", ""))
    return rocm_version
