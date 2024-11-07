import os
import sys

from cupy_builder import Context


def preflight_check(ctx: Context) -> bool:
    if sys.platform not in ('linux', 'win32'):
        print('Error: macOS is no longer supported', file=sys.stderr)
        return False

    source_root = ctx.source_root
    is_git = os.path.isdir(os.path.join(source_root, '.git'))
    for submodule in ('third_party/cccl',
                      'third_party/jitify',
                      'third_party/dlpack'):
        dirpath = os.path.join(source_root, submodule)
        if os.path.isdir(dirpath):
            if 0 < len(os.listdir(dirpath)):
                continue
        else:
            if not is_git:
                # sdist does not contain third_party directory
                continue

        if is_git:
            msg = f'''
===========================================================================
The directory {submodule} is a git submodule but is currently empty.
Please use the command:

    $ git submodule update --init

to populate the directory before building from source.
===========================================================================
        '''
        else:
            msg = f'''
===========================================================================
The directory {submodule} is a git submodule but is currently empty.
Instead of using ZIP/TAR archive downloaded from GitHub, use

    $ git clone --recursive https://github.com/cupy/cupy.git

to get a buildable CuPy source tree.
===========================================================================
        '''

        print(msg, file=sys.stderr)
        return False
    return True
