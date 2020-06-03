import contextlib

import optuna

from cupy.core import _optimize_config
from cupyx import time


def _optimize(optimize_config, target_func, suggest_func, default_best):
    assert isinstance(optimize_config, _optimize_config._OptimizationConfig)
    assert callable(target_func)
    assert callable(suggest_func)

    def objective(trial):
        args = suggest_func(trial)
        max_total_time = optimize_config.max_total_time_per_trial
        perf = time.repeat(target_func, args, max_duration=max_total_time)
        return perf.gpu_times.mean()

    study = optuna.create_study()
    study.enqueue_trial(default_best)
    study.optimize(
        objective,
        n_trials=optimize_config.max_trials,
        timeout=optimize_config.timeout)
    return study.best_trial


@contextlib.contextmanager
def optimize(*, key=None, **config_dict):
    """Context manager that optimizes kernel launch parameters.

    In this context, CuPy's routines find the best kernel launch parameter
    values (e.g., the number of threads and blocks). The found values are
    cached and reused with keys as the shapes, strides and dtypes of the
    given inputs arrays.

    Args:
        key (string or None): The cache key of optimizations.
        max_trials (int): The number of trials that defaults to 100.
        timeout (float):
            Stops study after the given number of seconds. Default is 1.
        max_total_time_per_trial (float):
            Repeats measuring the execution time of the routine for the
            given number of seconds. Default is 0.1.

    Examples
    --------
    >>> import cupy
    >>> from cupyx import optimizing
    >>>
    >>> x = cupy.arange(100)
    >>> with optimizing.optimize():
    ...     cupy.sum(x)
    ...
    array(4950)
    """
    old_context = _optimize_config.get_current_context()
    context = _optimize_config.get_new_context(key, _optimize, config_dict)
    _optimize_config.set_current_context(context)

    try:
        yield context
    finally:
        _optimize_config.set_current_context(old_context)
