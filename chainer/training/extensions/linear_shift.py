from __future__ import division

from chainer.training import extension


class LinearShift(extension.Extension):

    """Trainer extension to change an optimizer attribute linearly.

    This extension changes an optimizer attribute from the first value to the
    last value linearly within a specified duration. The typical use case is
    warming up of the momentum coefficient.

    For example, suppose that this extension is called at every iteration, and
    ``value_range == (x, y)`` and ``time_range == (i, j)``. Then, this
    extension keeps the attribute to be ``x`` up to the ``i``-th iteration,
    linearly shifts the value to ``y`` by the ``j``-th iteration, and then
    keeps the value to be ``y`` after the ``j``-th iteration.

    This extension is also called before the training loop starts by default.

    Args:
        attr (str): Name of the optimizer attribute to adjust.
        value_range (tuple of float): The first and the last values of the
            attribute.
        time_range (tuple of ints): The first and last counts of calls in which
            the attribute is adjusted.
        optimizer (~chainer.Optimizer): Target optimizer object. If it is None,
            the main optimizer of the trainer is used.

    """
    invoke_before_training = True

    def __init__(self, attr, value_range, time_range, optimizer=None):
        self._attr = attr
        self._value_range = value_range
        self._time_range = time_range
        self._optimizer = optimizer
        self._t = 0

    def __call__(self, trainer):
        optimizer = self._optimizer or trainer.updater.get_optimizer('main')
        t1, t2 = self._time_range
        v1, v2 = self._value_range

        if self._t <= t1:
            value = v1
        elif self._t >= t2:
            value = v2
        else:
            rate = (self._t - t1) / (t2 - t1)
            value = v1 + rate * (v2 - v1)
        setattr(optimizer, self._attr, value)

        self._t += 1

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
