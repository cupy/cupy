import time

from chainer.training import extension


def observe_value(key, target_func):
    """Returns a trainer extension to continuously record a value.

    Args:
        key (str): Key of observation to record.
        target_func (function): Function that returns the value to record.
            It must take one argument: :class:~chainer.training.Trainer object.
    Returns:
        The extension function.
    """
    @extension.make_extension(
        trigger=(1, 'epoch'), priority=extension.PRIORITY_WRITER)
    def _observe_value(trainer):
        trainer.observation[key] = target_func(trainer)
    return _observe_value


def observe_time(key='time'):
    """Returns a trainer extension to record the elapsed time.

    Args:
        key (str): Key of observation to record.

    Returns:
        The extension function.
    """
    start_time = time.time()
    return observe_value(key, lambda _: time.time() - start_time)


def observe_lr(optimizer, key='lr'):
    """Returns a trainer extension to record the learning rate.

    Args:
        optimizer (~chainer.Optimizer): Optimizer object whose learning rate is recorded.
        key (str): Key of observation to record.

    Returns:
        The extension function.
    """
    return observe_value(key, lambda _: optimizer.lr)
