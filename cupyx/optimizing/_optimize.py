import contextlib
import math

import optuna

from cupy.core import _optimize_config
from cupy.cuda import stream as stream_module


def _optimize(optimize_config, target_func, suggest_func):
    assert isinstance(optimize_config, _optimize_config._OptimizationConfig)
    assert callable(target_func)
    assert callable(suggest_func)

    def objective(trial):

        args = suggest_func(trial)

        def _measure(n):
            # Returns total GPU time (in seconds)
            stream = stream_module.get_current_stream()
            stream.synchronize()
            ev1 = stream_module.Event()
            ev2 = stream_module.Event()
            ev1.synchronize()
            ev1.record()

            for _ in range(n):
                target_func(*args)

            ev2.record()
            ev2.synchronize()
            time = stream_module.get_elapsed_time(ev1, ev2) * 1e-3
            return time

        min_total_time = optimize_config.min_total_time_per_trial
        expected_total_time = optimize_config.expected_total_time_per_trial

        _measure(1)  # warmup

        n = 1
        while True:
            total_time = _measure(n)
            if total_time > min_total_time:
                break
            n = max(
                n+1,
                int(math.ceil(expected_total_time * n / total_time)))

        return total_time / n

    study = optuna.create_study()
    study.optimize(
        objective,
        n_trials=optimize_config.max_trials,
        timeout=optimize_config.timeout)
    return study.best_trial


@contextlib.contextmanager
def optimize(*, key=None, **config_dict):
    assert key is not None

    old_context = _optimize_config.get_current_context()
    context = _optimize_config.get_new_context(key, _optimize, config_dict)
    thread_local = _optimize_config._thread_local
    thread_local.current_context = context

    try:
        yield context
    finally:
        thread_local.current_context = old_context
