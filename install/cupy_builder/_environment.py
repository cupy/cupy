import functools
import glob
import os
import sys
from typing import Any, Callable, Optional


def _memoize(f: Callable) -> Callable:
    memo = {}

    @functools.wraps(f)
    def ret(*args: Any, **kwargs: Any) -> Any:
        key = (args, frozenset(kwargs.items()))
        if key not in memo:
            memo[key] = f(*args, **kwargs)
        return memo[key]
    return ret


@_memoize
def get_nvtx_path() -> Optional[str]:
    assert sys.platform == 'win32'

    prog = os.environ.get('ProgramFiles', 'C:\\Program Files')
    pattern = os.path.join(
        prog, 'NVIDIA Corporation', 'Nsight Systems *', 'target-windows-x64',
        'nvtx',
    )
    print(f'Looking for NVTX: {pattern}')
    candidates = sorted(glob.glob(pattern))
    if len(candidates) != 0:
        # Pick the latest one
        nvtx = candidates[-1]
        print(f'Using NVTX at: {nvtx}')
        return nvtx
    if os.environ.get('CONDA_BUILD', '0') == '1':
        return os.environ['PREFIX']
    print('NVTX could not be found')
    return None
