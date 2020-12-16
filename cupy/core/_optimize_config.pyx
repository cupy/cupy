import pickle
import threading


cdef _thread_local = threading.local()
cdef _contexts = {}


cdef class _OptimizationConfig:

    def __init__(
            self, optimize_impl, *,
            int max_trials=100,
            float timeout=1,
            float expected_total_time_per_trial=100 * 1e-6,
            float max_total_time_per_trial=0.1):
        self.optimize_impl = optimize_impl
        self.max_trials = max_trials
        self.timeout = timeout
        self.expected_total_time_per_trial = expected_total_time_per_trial
        self.max_total_time_per_trial = max_total_time_per_trial


cdef class _OptimizationContext:

    def __init__(self, str key, _OptimizationConfig config):
        self.key = key
        self.config = config
        self._params_map = {}
        self._dirty = False

    def get_params(self, key):
        return self._params_map.get(key)

    def set_params(self, key, params):
        self._params_map[key] = params
        self._dirty = True

    def save(self, filepath):
        with open(filepath, mode='wb') as f:
            pickle.dump((self.key, self._params_map), f)
        self._dirty = False

    def load(self, filepath):
        with open(filepath, mode='rb') as f:
            key, params_map = pickle.load(f)
        if key != self.key:
            raise ValueError(
                'Optimization key mismatch {} != {}'.format(key, self.key))
        self._params_map = params_map
        self._dirty = False

    def _is_dirty(self):
        return self._dirty


cpdef _OptimizationContext get_current_context():
    try:
        return _thread_local.current_context
    except AttributeError:
        return None


def set_current_context(_OptimizationContext context):
    _thread_local.current_context = context


def get_new_context(
        str key, object optimize_impl, dict config_dict):
    c = _contexts.get(key)
    if c is None:
        config = _OptimizationConfig(optimize_impl, **config_dict)
        c = _OptimizationContext(key, config)
        _contexts[key] = c
    return c


def _clear_all_contexts_cache():
    global _contexts
    assert get_current_context() is None
    _contexts = {}
