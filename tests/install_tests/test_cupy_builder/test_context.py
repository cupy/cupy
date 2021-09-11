from cupy_builder._context import _get_env_bool, Context


class TestGetEnvBool:
    def test_true(self):
        assert _get_env_bool('V', True, {})
        assert _get_env_bool('V', True, {'V': '1'})
        assert _get_env_bool('V', False, {'V': '1'})

    def test_false(self):
        assert not _get_env_bool('V', False, {})
        assert not _get_env_bool('V', False, {'V': '0'})
        assert not _get_env_bool('V', True, {'V': '0'})


class TestContext:
    def test_default(self):
        ctx = Context('.', _env={})
        assert ctx.enable_thrust
        assert not ctx.use_cuda_python
        assert not ctx.use_hip
