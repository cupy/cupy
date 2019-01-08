import numpy


try:
    _AxisError = numpy.AxisError
except AttributeError:
    class IndexOrValueError(IndexError, ValueError):

        def __init__(self, *args, **kwargs):
            super(IndexOrValueError, self).__init__(*args, **kwargs)

    _AxisError = IndexOrValueError
