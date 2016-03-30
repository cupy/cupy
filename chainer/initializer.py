import numpy as np


# Original code forked from MIT licensed keras project
# https://github.com/fchollet/keras/blob/master/keras/initializations.py

class Initialzer(object):

    def __call__(self, shape):
        NotImplementedError()


def get_fans(shape):
    fan_in = np.prod(shape[1:])
    fan_out = shape[0]
    return fan_in, fan_out
