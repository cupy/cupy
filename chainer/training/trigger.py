class IntervalTrigger(object):

    """Trigger based on a fixed interval.

    This trigger accepts iterations divided by a given interval. There are two
    ways to specify the interval: per iterations and epochs. `Iteration` means
    the number of updates, while `epoch` means the number of sweeps over the
    training dataset. Both values are defined by the updater.

    For the description of triggers, see :func:`get_trigger`.

    Args:
        period (int): Length of the interval.
        unit (str): Unit of the length specified by ``period``. It must be
            either ``'iteration'`` or ``'epoch'``.

    """

    def __init__(self, period, unit):
        self.period = period
        assert unit == 'epoch' or unit == 'iteration'
        self.unit = unit
        self.count = 0

    def __call__(self, trainer):
        """Decides whether the extension should be called on this iteration.

        In order to support non-integer epoch intervals, it does not rely on
        ``updater.iterator.is_new_epoch``.

        Args:
            trainer (Trainer): Trainer object that this trigger is associated
                with. The updater associated with this trainer is used to
                determine if the trigger should fire.

        Returns:
            bool: True if the corresponding extension should be invoked in this
                iteration.

        """
        updater = trainer.updater
        if self.unit == 'epoch':
            prev = self.count
            self.count = updater.epoch_detail // self.period
            return prev != self.count
        else:
            iteration = updater.iteration
            return iteration > 0 and iteration % self.period == 0


def get_trigger(trigger):
    """Gets a trigger object.

    Trigger object is a callable that accepts a
    :class:`~chainer.training.Trainer` object as an argument and returns a
    boolean value. When it returns True, various kinds of events can occur
    depending on the context in which the trigger is used. For example, if the
    trigger is passed to the :class:`~chainer.training.Trainer` as the `stop
    trigger`, the training loop breaks when the trigger returns True. If the
    trigger is passed to the :meth:`~chainer.training.Trainer.extend` method of
    a trainer, then the registered extension is invoked only when the trigger
    returns True.

    This function returns a trigger object based on the argument.
    If ``trigger`` is already a callable, it just returns the trigger. If
    ``trigger`` is None, it returns a trigger that never fires. Otherwise, it
    passes the value to :class:`IntervalTrigger`.

    Args:
        trigger: Trigger object. It can be either an already built trigger
            object (i.e., a callable object that accepts a trainer object and
            returns a bool value), or a tuple. In latter case, the tuple is
            passed to :class:`IntervalTrigger`.

    Returns:
        ``trigger`` if it is a callable, otherwise a :class:`IntervalTrigger`
        object made from ``trigger``.

    """
    if callable(trigger):
        return trigger
    elif trigger is None:
        return _never_fire_trigger
    else:
        return IntervalTrigger(*trigger)


def _never_fire_trigger(trainer):
    return False
