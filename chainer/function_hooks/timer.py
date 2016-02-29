import time

import numpy

from chainer import cuda
from chainer import function


class TimerHook(function.FunctionHook):
    """Function hook for measuring elapsed time of functions.

    Attributes:
        call_history: List of measurement results. It consists of pairs of
            the function that calls this hook and the elapsed time.
    """

    name = 'TimerHook'

    def __init__(self):
        self.call_history = []

    def preprocess(self, function, in_data, out_grad=None):
        xp = cuda.get_array_module(*in_data)
        if xp == numpy:
            self.start = time.time()
        else:
            self.start = cuda.Event()
            self.stop = cuda.Event()
            self.start.record()

    def postprocess(self, function, in_data, out_grad=None):
        xp = cuda.get_array_module(*in_data)
        if xp == numpy:
            self.stop = time.time()
            elapsed_time = self.stop - self.start
        else:
            self.stop.record()
            self.stop.synchronize()
            elapsed_time = cuda.cupy.cuda.get_elapsed_time(
                self.start, self.stop)
        self.call_history.append((function, elapsed_time))

    def total_time(self):
        """Returns total elapsed time."""
        return sum(t for (_, t) in self.call_history)
