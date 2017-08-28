import os
import subprocess
import sys


def run_example(path, *args):
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
