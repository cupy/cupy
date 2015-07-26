import numpy

import cupy


def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    return numpy.array_repr(cupy.asnumpy(arr), max_line_width, precision,
                            suppress_small)


def array_str(arr, max_line_width=None, precision=None, suppress_small=None):
    return numpy.array_str(cupy.asnumpy(arr), max_line_width, precision,
                           suppress_small)
