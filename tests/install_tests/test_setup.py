from __future__ import annotations

import multiprocessing
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('start_method', ['spawn', 'forkserver'])
def test_parallel_cythonize_setup(tmp_path, start_method):
    pytest.importorskip('Cython.Build')
    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f'{start_method} is unavailable')

    setup_path = Path(__file__).resolve().parents[2] / 'setup.py'
    sources = []
    for name in ('first', 'second'):
        source = tmp_path / f'{name}.pyx'
        source.write_text('def value():\n    return 1\n')
        sources.append(str(source))

    # Reproduce the setup.py entry point seen by Cython worker processes.
    # Workers re-import it as __mp_main__, which must not start another build.
    code = '''
import __main__
import multiprocessing
import sys
from Cython.Build import cythonize

__main__.__file__ = sys.argv[1]
sys.argv = [sys.argv[1], 'build']
multiprocessing.set_start_method(START_METHOD)
cythonize(SOURCES, nthreads=2, language_level=3)
'''.replace('START_METHOD', repr(start_method)).replace('SOURCES', repr(sources))
    result = subprocess.run(
        [sys.executable, '-c', code, str(setup_path)],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    for source in sources:
        assert Path(source).with_suffix('.c').is_file()
