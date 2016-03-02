from __future__ import print_function
import sys

import chainer
from chainer import cuda
from chainer import function


class PrintHook(function.FunctionHook):
    """Function hook that prints debug information.

    Attributes are same as the keyword argument of print function.

    Attributes:
        sep: Separator of print function.
        end: Character to be added at the end of print function.
        file: Output file_like object that that redirect to.
        flush: If ``True``, print function forcibly flushes the text stream.
    """

    name = 'PrintHook'

    def __init__(self, sep='', end='\n', file=sys.stdout):
        self.sep = sep
        self.end = end
        self.file = file

    def _print(self, msg):
        print(msg, sep=self.sep, end=self.end, file=self.file)

    def _process(self, function, in_data, out_grad=None):
        self._print('function\t{}'.format(function.label))
        self._print('input data')
        for d in in_data:
            self._print(chainer.Variable(d).debug_print())
        if out_grad is not None:
            self._print('output gradient')
            for d in out_grad:
                xp = cuda.get_array_module(d)
                v = chainer.Variable(xp.empty_like(d, dtype=d.dtype))
                v.grad = d
                self._print(v.debug_print())

    def forward_preprocess(self, function, in_data):
        self._process(function, in_data)

    def backward_preprocess(self, function, in_data, out_grad):
        self._process(function, in_data, out_grad)

    def forward_postprocess(self, function, in_data):
        self._process(function, in_data)

    def backward_postprocess(self, function, in_data, out_grad):
        self._process(function, in_data, out_grad)
