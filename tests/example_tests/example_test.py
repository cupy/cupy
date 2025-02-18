import os
import subprocess
import sys

import cupy


def run_example(path, *args):
    # Free memory occupied in the main process before launching an example.
    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    examples_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'examples')
    fullpath = os.path.join(examples_path, path)

    try:
        return subprocess.check_output(
            (sys.executable, fullpath) + args,
            stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print('Original error message:')
        print(e.output.decode('utf-8'))
        raise
