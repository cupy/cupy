import os
import typing


def _get_env_bool(name: str, default: bool, env: typing.Mapping) -> bool:
    return env[name] != '0' if name in env else default


class Context:
    def __init__(
            self, source_root: str, *, _env: typing.Mapping = os.environ):
        self.source_root = source_root

        self.enable_thrust = _get_env_bool(
            'CUPY_SETUP_ENABLE_THRUST', True, _env)
        self.use_cuda_python = _get_env_bool(
            'CUPY_USE_CUDA_PYTHON', False, _env)
        self.use_hip = _get_env_bool(
            'CUPY_INSTALL_USE_HIP', False, _env)
