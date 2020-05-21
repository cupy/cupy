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


class _OptimizationConfig:

    def __init__(
            self, optimize_impl, *,
            max_trials=100,
            timeout=1,
            expected_total_time_per_trial=100 * 1e-6,
            max_total_time_per_trial=90 * 1e-6):
        self.optimize_impl = optimize_impl
        self.max_trials = max_trials
        self.timeout = timeout
        self.expected_total_time_per_trial = expected_total_time_per_trial
        self.max_total_time_per_trial = max_total_time_per_trial


def get_current_context():
    try:
        return _thread_local.current_context
    except AttributeError:
        return None


def get_new_context(key, optimize_impl, config_dict):
    c = _contexts.get(key)
    if c is None:
        config = _OptimizationConfig(optimize_impl, **config_dict)
        c = _OptimizationContext(key, config)
        _contexts[key] = c
    return c
