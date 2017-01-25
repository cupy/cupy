class IrregularTrigger(object):

    """Trigger invoked when specific value reaches the indicated values.

    This trigger accepts iterations indicated by given periods. There are two
    ways to specify the preriods: iterations and epochs. `Iteration` means
    the number of updates, while `epoch` means the number of sweeps over the
    training dataset. Fractional values are allowed if the period is a
    number of epochs; the trigger uses the `iteration` and `epoch_detail`
    attributes defined by the updater.

    Args:
        periods (int or float or list of int or float): Length of the interval.
            Must be an integer or list of integer if unit is ``'iteration'``.
        unit (str): Unit of the length specified by ``period``. It must be
            either ``'iteration'`` or ``'epoch'``.

    """

    def __init__(self, period, unit):
        self.period = period if isinstance(period, list) else [period]
        assert unit == 'epoch' or unit == 'iteration'
        self.unit = unit
        self.count = [0 for _ in range(len(self.period))]
        self.valid = [True for _ in range(len(self.period))]

    def __call__(self, trainer):
        """Decides whether the extension should be called on this iteration.

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
            self.count = [updater.epoch_detail // p for p in self.period]
            flag = [p != c for p, c in zip(prev, self.count)]
            ans = any([f and v for f, v in zip(flag, self.valid)])
            self.valid = [not f and v for f, v in zip(flag, self.valid)]
            return ans
        else:
            iteration = updater.iteration
            return iteration > 0 and iteration in self.period
