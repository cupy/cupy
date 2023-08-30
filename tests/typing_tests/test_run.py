from pathlib import Path


def generate_test(_t: Path) -> None:
    assert not _t.name.endswith("._numpy.pyi")
    name = str(_t.relative_to(Path(__file__).parent))
    name = name.replace("_tests/", "/").replace("/test_", "/")
    name = name.replace("/", "_").replace(".pyi", "")

    with open(_t) as f:
        lines = f.readlines()

    def test() -> None:
        for _lineno, _line in enumerate(lines, start=1):
            if not "# E: " in _line:
                try:
                    exec(_line)
                except Exception:
                    print(f"{_t.stem}:{_lineno} {_line}")
                    raise

    globals()[f"test_{name}"] = test


for t in Path(__file__).parent.glob("**/*.pyi"):
    generate_test(t)
