class ManualScheduleTrigger(object):

    """Trigger invoked at specified point(s) of iterations or epochs.

    This trigger accepts iterations or epochs indicated by given point(s).
    There are two ways to specify the point(s): iteration and epoch.
    `iteration` means the number of updates, while `epoch` means the number
    of sweeps over the training dataset. Fractional values are allowed
    if the point is a number of epochs; the trigger uses the `iteration`
    and `epoch_detail` attributes defined by the updater.

    Args:
        points (int, float, or list of int or float): time of the trigger.
            Must be an integer or list of integer if unit is ``'iteration'``.
        unit (str): Unit of the time specified by ``points``. It must be
            either ``'iteration'`` or ``'epoch'``.

    """

    def __init__(self, points, unit):
        self.points = points if isinstance(points, list) else [points]
        assert unit == 'epoch' or unit == 'iteration'
        self.unit = unit
        self.count = [0 for _ in range(len(self.points))]
        self.valid = [True for _ in range(len(self.points))]

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
            self.count = [updater.epoch_detail // p for p in self.points]
            flag = [p != c for p, c in zip(prev, self.count)]
            ans = any([f and v for f, v in zip(flag, self.valid)])
            self.valid = [not f and v for f, v in zip(flag, self.valid)]
            return ans
        else:
            iteration = updater.iteration
            return iteration > 0 and iteration in self.points
