import dataclasses
from pathlib import Path
import shutil
import tempfile
import typing
import re

import pytest

try:
    import mypy.api
    NO_MYPY = False
except ImportError:
    NO_MYPY = True


class Pos(typing.NamedTuple):
    filename: str
    lineno: int


@dataclasses.dataclass
class MypyError:
    typ: str
    mes: str


def generate_test_code(tmpdir: Path) -> None:
    # Generate test for numpy compatibility
    for t in tmpdir.glob("**/*.pyi"):
        assert not t.name.endswith("._numpy.pyi")
        with open(t) as f:
            text = f.read()

    # Generate a test code with "import numpy as xp"
    if "import cupy as xp\n" in text:
        name = t.name.replace(".pyi", "_numpy.pyi")
        text = text.replace("import cupy as xp\n", "import numpy as xp\n")
        with (t.parent / name).open("w") as f:
            f.write(text)

    # Check if "mypy.ini" exists
    assert (tmpdir / "typing_tests" / "mypy.ini").exists()


def collect_xfails(tests: list[Path]) -> dict[Pos, MypyError]:
    xfails: dict[Pos, MypyError] = {}
    for t in tests:
        with open(t) as f:
            for i, text in enumerate(f.readlines()):
                pos = Pos(str(t), i + 1)
                if "# E:" in text:
                    m = re.match(r".*# E:(?P<mes>.*)\[(?P<type>.+)\]", text)
                    assert m is not None, f"{text} (Hint: Missing Error code?)"
                    xfails[pos] = MypyError(m.group(2), m.group(1).strip())
    return xfails


def run_mypy(tests: list[Path]) -> dict[Pos, MypyError]:
    result = mypy.api.run([str(t) for t in tests])
    mypy_fails: dict[Pos, MypyError] = {}
    for line in result[0].strip().split("\n"):
        if re.match(
                r"^Found (\d)+ error(s?) in (\d)+ file(s?) "
                r"\(checked (\d)+ source file(s?)\)$", line):
            continue
        m = re.match(r"^(.+):(\d+): (error|note): (.+)$", line)
        assert m is not None, line
        pos = Pos(m.group(1), int(m.group(2)))
        text = m.group(4)
        if pos in mypy_fails:
            mypy_fails[pos].mes = mypy_fails[pos].mes + "\n" + text
        else:
            m = re.match(r"^(.+)\[(.+)\]$", text)
            assert m is not None
            mypy_fails[pos] = MypyError(m.group(2), m.group(1))
    return mypy_fails


@pytest.mark.skipif(NO_MYPY, reason="mypy is not installed")
def test_typecheck() -> None:
    test_dir = Path(__file__).parent

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copytree(
            test_dir, tmpdir + "/typing_tests",
            ignore=shutil.ignore_patterns("*.py", "_numpy.pyi", ".gitignore"),
        )

        generate_test_code(Path(tmpdir))
        tests = list(Path(tmpdir).glob("**/*.pyi"))
        xpass = collect_xfails(tests)
        mypy_fails = run_mypy(tests)

    fails: dict[Pos, MypyError] = {}
    invalid: dict[Pos, MypyError] = {}

    # Check result
    for pos, err in mypy_fails.items():
        if pos in xpass:
            xfail = xpass.pop(pos)
            if pos.filename.endswith("_numpy.pyi"):
                continue  # Skips checks of compatibility of error messages
            if xfail.mes not in err.mes or xfail.typ != err.typ:
                invalid[pos] = err
        else:
            fails[pos] = err

    if invalid or fails or xpass:
        # Show error messages
        message = "\n\n"
        if invalid:
            message += "Invalid error message:\n"
            for (name, lineno), err in invalid.items():
                message += f"{name}:{lineno}: {err.mes} [{err.typ}]\n"
        if fails:
            message += "FAILS:\n"
            for (name, lineno), err in fails.items():
                message += f"{name}:{lineno}: {err.mes} [{err.typ}]\n"
        if xpass:
            message += "XPASS:\n"
            for (name, lineno), err in xpass.items():
                message += f"{name}:{lineno}: {err.mes} [{err.typ}]\n"
        pytest.fail(message)
