from pathlib import Path

import pytest


@pytest.mark.parametrize("t", map(str, Path(__file__).parent.glob("**/*.pyi")))
def test_run(t: Path) -> None:
    assert not t.endswith("._numpy.pyi")

    with open(t) as f:
        lines = f.readlines()

    exec_locals = {}
    for _lineno, _line in enumerate(lines, start=1):
        if "# E: " not in _line:
            try:
                exec(_line, None, exec_locals)
            except Exception:
                print(f"{t}:{_lineno} {_line}")
                raise
