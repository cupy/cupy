import contextlib
import math


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
def optimize(*, key=None, **config_dict):
    if not _optuna_available:
        raise RuntimeError(
            'Optuna is required to run optimization. '
            'See https://optuna.org/ for the installation instructions.')

    old_context = _optimize_config.get_current_context()
    context = _optimize_config.get_new_context(key, _optimize, config_dict)
    _optimize_config.set_current_context(context)

    try:
        yield context
    finally:
        _optimize_config.set_current_context(old_context)
