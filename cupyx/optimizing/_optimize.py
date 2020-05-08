from cupy.core import _optimize_config
import contextlib


@contextlib.contextmanager
def optimize(*, key=None, **config_dict):
    assert key is not None

    old_context = _optimize_config.get_current_context()
    context = _optimize_config.get_new_context(key, config_dict)
    thread_local = _optimize_config._thread_local
    thread_local.current_context = context

    try:
        yield context
    finally:
        thread_local.current_context = old_context
