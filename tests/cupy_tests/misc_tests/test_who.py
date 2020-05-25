import cupy


class TestWho:
    def test_who_empty(self, capsys):
        cupy.who()
        out, err = capsys.readouterr()
        lines = out.split("\n")
        assert len(lines) == 3
        assert lines[1] == "Upper bound on total bytes  =       0"

    def test_who_local_var(self, capsys):
        # Variables declared inside an object function are not visible
        # this is true also for numpy
        x = cupy.ones(10)  # NOQA
        cupy.who()
        out, err = capsys.readouterr()
        lines = out.split("\n")
        assert len(lines) == 3
        assert lines[1] == "Upper bound on total bytes  =       0"

    def test_who_global(self, capsys):
        global x
        x = cupy.ones(10)  # NOQA
        cupy.who()
        out, err = capsys.readouterr()
        lines = out.split("\n")
        assert lines[-4].split() == ["x", "10", "80", "float64"]
        assert lines[-2] == "Upper bound on total bytes  =       80"

    def test_who_global_multi(self, capsys):
        global x
        global y
        x = cupy.ones(10)  # NOQA
        y = cupy.ones(20, dtype=cupy.int32)  # NOQA
        cupy.who()
        out, err = capsys.readouterr()
        lines = out.split("\n")
        assert lines[-5].split() == ["x", "10", "80", "float64"]
        assert lines[-4].split() == ["y", "20", "80", "int32"]
        assert lines[-2] == "Upper bound on total bytes  =       160"

    def test_who_dict_arrays(self, capsys):
        var_dict = {"x": cupy.ones(10)}
        cupy.who(var_dict)
        out, err = capsys.readouterr()
        lines = out.split("\n")
        assert lines[-4].split() == ["x", "10", "80", "float64"]
        assert lines[-2] == "Upper bound on total bytes  =       80"

    def test_who_dict_empty(self, capsys):
        global x
        x = cupy.ones(10)  # NOQA
        cupy.who({})
        out, err = capsys.readouterr()
        lines = out.split("\n")
        assert lines[-2] == "Upper bound on total bytes  =       0"
