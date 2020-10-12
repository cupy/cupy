import contextlib
import math
import os
import warnings


try:
    import optuna
    _optuna_available = True
except ImportError:
    _optuna_available = False


from cupy.core import _optimize_config
from cupyx import time


def _optimize(
        optimize_config, target_func, suggest_func,
        default_best, ignore_error=()):
    assert isinstance(optimize_config, _optimize_config._OptimizationConfig)
    assert callable(target_func)
    assert callable(suggest_func)

    def objective(trial):
        args = suggest_func(trial)
        max_total_time = optimize_config.max_total_time_per_trial
        try:
            perf = time.repeat(target_func, args, max_duration=max_total_time)
            return perf.gpu_times.mean()
        except Exception as e:
            if isinstance(e, ignore_error):
                return math.inf
            else:
                raise e

    study = optuna.create_study()
    study.enqueue_trial(default_best)
    study.optimize(
        objective,
        n_trials=optimize_config.max_trials,
        timeout=optimize_config.timeout)
    return study.best_trial


@contextlib.contextmanager
def optimize(*, key=None, path=None, readonly=False, **config_dict):
    """Context manager that optimizes kernel launch parameters.

    In this context, CuPy's routines find the best kernel launch parameter
    values (e.g., the number of threads and blocks). The found values are
    cached and reused with keys as the shapes, strides and dtypes of the
    given inputs arrays.

    Args:
        key (string or None): The cache key of optimizations.
        path (string or None): The path to save optimization cache records.
            When path is specified and exists, records will be loaded from
            the path. When readonly option is set to ``False``, optimization
            cache records will be saved to the path after the optimization.
        readonly (bool): See the description of ``path`` option.
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

    .. note::
      Optuna (https://optuna.org) installation is required.
      Currently it works for reduction operations only.
    """
    if not _optuna_available:
        raise RuntimeError(
            'Optuna is required to run optimization. '
            'See https://optuna.org/ for the installation instructions.')

    old_context = _optimize_config.get_current_context()
    context = _optimize_config.get_new_context(key, _optimize, config_dict)
    _optimize_config.set_current_context(context)

    if path is not None:
        if os.path.exists(path):
            context.load(path)
        elif readonly:
            warnings.warn('''
The specified path {} could not be found, and the readonly option is set.
The optimization results will never be stored.
'''.format(path))

    try:
        yield context
        if path is not None and not readonly and context._is_dirty():
            context.save(path)
    finally:
        _optimize_config.set_current_context(old_context)
