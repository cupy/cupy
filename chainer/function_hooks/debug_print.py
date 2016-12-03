from __future__ import print_function
import sys

import chainer
from chainer import cuda
from chainer import function


class PrintHook(function.FunctionHook):
    """Function hook that prints debug information.

    This function hook outputs the debug information of input arguments of
    ``forward`` and ``backward`` methods involved in the hooked functions
    at preprocessing time (that is, just before each method is called).

    The basic usage is to use it with ``with`` statement.

    >>> import chainer, chainer.functions as F, chainer.links as L
    ... from chainer import function_hooks
    ... l = L.Linear(10, 10)
    ... x = chainer.Variable(numpy.zeros((1, 10), 'f'))
    ... with function_hooks.PrintHook():
    ...     y = l(x)
    ...     z = F.sum(y)
    ...     z.backward()

    In this example, ``PrintHook`` shows the debug information of
    forward propagation of ``LinearFunction`` (which is implicitly
    called by ``l``) and ``Sum`` (called by ``F.sum``)
    and backward propagation of ``z`` and ``y``.

    Unlike simple "debug print" technique, where users insert print functions
    at every function to be inspected, we can show the information
    of all functions involved with single ``with`` statement.

    Further, this hook enables us to show the information of
    ``backward`` methods without inserting print functions into
    Chainer's library code.

    Attributes:
        sep: Separator of print function.
        end: Character to be added at the end of print function.
        file: Output file_like object that that redirect to.
        flush: If ``True``, this hook forcibly flushes the text stream
            at the end of preprocessing.
    """

    name = 'PrintHook'

    def __init__(self, sep='', end='\n', file=sys.stdout, flush=True):
        self.sep = sep
        self.end = end
        self.file = file
        self.flush = flush

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
                v = chainer.Variable(xp.zeros_like(d, dtype=d.dtype))
                v.grad = d
                self._print(v.debug_print())
        if self.flush:
            self.file.flush()

    def forward_preprocess(self, function, in_data):
        self._process(function, in_data)

    def backward_preprocess(self, function, in_data, out_grad):
        self._process(function, in_data, out_grad)
