from pathlib import Path
import re

import pytest

try:
    import mypy.api
    NO_MYPY = False
except ImportError:
    NO_MYPY = True


@pytest.mark.skipif(NO_MYPY, reason="mypy is not installed")
def test_typecheck() -> None:
    test_dir = Path(__file__).parent

    # Generate test for numpy compatibility
    for t in test_dir.glob("**/*[!_numpy].pyi"):
        assert t.name.endswith(".pyi")
        assert not t.name.endswith("._numpy.pyi")
        with open(t) as f:
            text = f.read()

        # Generate a test code with "import numpy as xp"
        if "import cupy as xp\n" in text:
            name = t.name.replace(".pyi", "_numpy.pyi")
            text = text.replace("import cupy as xp\n", "import numpy as xp\n")
            with (t.parent / name).open("w") as f:
                f.write(text)

    xpass: dict[tuple[str, str], tuple[str, str]] = {}
    fails: dict[tuple[str, str], tuple[str, str]] = {}
    invalid: dict[tuple[str, str], tuple[str, str]] = {}

    # Collect expected errors
    tests = list(test_dir.glob("**/*.pyi"))
    for t in tests:
        name = str(t.relative_to(Path.cwd()))
        with open(t) as f:
            for i, text in enumerate(f.readlines()):
                lineno = str(i + 1)
                if "# E:" in text:
                    m = re.match(r".*# E:(?P<mes>.*)\[(?P<type>.+)\]", text)
                    assert m is not None, f"{text} (Hint: Missing Error code?)"
                    mes, typ = m.groups()
                    xpass[(name, lineno)] = (mes.strip(), typ)

    # Run mypy
    result = mypy.api.run([str(t) for t in tests])
    mypy_fails: dict[tuple[str, str], tuple[str, str, list[str]]] = {}
    for line in result[0].strip().split("\n"):
        if re.match(
                r"^Found (\d)+ error(s?) in (\d)+ file(s?) "
                r"\(checked (\d)+ source file(s?)\)$", line):
            continue
        m = re.match(r"^(.+):(\d+): (error|note): (.+)$", line)
        assert m is not None, line
        name, lineno, error_note, text = m.groups()
        if (name, lineno) in mypy_fails:
            mypy_fails[(name, lineno)][2].append(text)
        else:
            m = re.match(r"^(.+)\[(.+)\]$", text)
            assert m is not None
            mes, typ = m.groups()
            mypy_fails[(name, lineno)] = (error_note, typ, [mes])

    # Check result
    for (name, lineno), (error_note, typ, mes_list) in mypy_fails.items():
        mes = '\n'.join(mes_list)
        if (name, lineno) in xpass:
            emes, etyp = xpass.pop((name, lineno))
            if name.endswith("_numpy.pyi"):
                continue  # Skips checks of compatibility of error messages
            if emes not in mes or typ != etyp:
                invalid[(name, lineno)] = (mes, typ)
        else:
            fails[(name, lineno)] = (mes, typ)

    if invalid or fails or xpass:
        message = "\n\n"
        if invalid:
            message += "Invalid error message:\n"
            for (name, lineno), (mes, typ) in invalid.items():
                message += f"{name}:{lineno}: {mes} [{typ}]\n"
        if fails:
            message += "FAILS:\n"
            for (name, lineno), (mes, typ) in fails.items():
                message += f"{name}:{lineno}: {mes} [{typ}]\n"
        if xpass:
            message += "XPASS:\n"
            for (name, lineno), (mes, typ) in xpass.items():
                message += f"{name}:{lineno}: {mes} [{typ}]\n"
        pytest.fail(message)
