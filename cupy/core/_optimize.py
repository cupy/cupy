import contextlib
import threading


_thread_local = threading.local()


_contexts = {}


class _Context:
    def __init__(self, key):
        self.key = key
        self.params = None

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params


def _get_context(key):
    c = _contexts.get(key)
    if c is None:
        c = _Context(key)
        _contexts[key] = c
    return c


def _get_current_context():
    try:
        return _thread_local.current_context
    except AttributeError:
        return None


@contextlib.contextmanager
def optimize(*, key=None):
    assert key is not None

    old_context = _get_current_context()
    _thread_local.current_context = _get_context(key)

    try:
        yield
    finally:
        _thread_local.current_context = old_context
