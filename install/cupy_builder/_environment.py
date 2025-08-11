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

    if os.environ.get('CONDA_BUILD', '0') == '1':
        return os.environ['PREFIX']

    candidates = []

    # $Env:ProgramFiles\NVIDIA Corporation\Nsight Systems 2025.3.2\
    # target-windows-x64\nvtx\include\nvtx3\nvToolsExt.h
    prog = os.environ.get('ProgramFiles', 'C:\\Program Files')
    pattern = os.path.join(
        prog, 'NVIDIA Corporation', 'Nsight Systems *', 'target-windows-x64',
        'nvtx',
    )
    candidates += sorted(p for p in glob.glob(pattern) if os.path.exists(p))

    # $Env:CUDA_PATH\include\nvtx3\nvToolsExt.h
    cuda_path = os.environ.get('CUDA_PATH', None)
    if cuda_path is not None:
        cuda_include_path = os.path.join(cuda_path, 'include')
        if os.path.exists(cuda_include_path):
            candidates += [cuda_include_path]

    print(f'Looking for NVTX from: {candidates}')
    if len(candidates) != 0:
        # Pick the one from CUDA_PATH or use the latest one from Nsight
        nvtx = candidates[-1]
        print(f'Using NVTX at: {nvtx}')
        return nvtx
    print('NVTX could not be found')
    return None
