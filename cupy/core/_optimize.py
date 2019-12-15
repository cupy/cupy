import contextlib
import threading


_thread_local = threading.local()


_contexts = {}


class _OptimizationContext:
    def __init__(self, key, config):
        self.key = key
        self.config = config
        self.params_map = {}

    def get_params(self, key):
        return self.params_map.get(key)

    def set_params(self, key, params):
        self.params_map[key] = params


def _get_context(key, config_dict):
    c = _contexts.get(key)
    if c is None:
        config = _OptimizationConfig(**config_dict)
        c = _OptimizationContext(key, config)
        _contexts[key] = c
    return c


class _OptimizationConfig:

    def __init__(
            self, *,
            max_trials=100,
            timeout=1,
            expected_total_time_per_trial=100 * 1e-6,
            min_total_time_per_trial=90 * 1e-6,
    ):
        self.max_trials = max_trials
        self.timeout = timeout
        self.expected_total_time_per_trial = expected_total_time_per_trial
        self.min_total_time_per_trial = min_total_time_per_trial


def _get_current_context():
    try:
        return _thread_local.current_context
    except AttributeError:
        return None


@contextlib.contextmanager
def optimize(*, key=None, **config_dict):
    assert key is not None

    old_context = _get_current_context()
    context = _get_context(key, config_dict)
    _thread_local.current_context = context

    try:
        yield context
    finally:
        _thread_local.current_context = old_context
