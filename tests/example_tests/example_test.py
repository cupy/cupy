import os
import subprocess
import sys


def run_example(path, *args):
    examples_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'examples')
    fullpath = os.path.join(examples_path, path)

    return subprocess.check_output((sys.executable, fullpath) + args)
