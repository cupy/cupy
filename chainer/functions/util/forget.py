from chainer import cuda
from chainer import function
from chainer import variable


class _DummyFunction(function.Function):

    def __init__(self, grads):
        self.grads = grads

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        return xp.array(0),

    def backward(self, inputs, outputs):
        return self.grads


class Forget(function.Function):

    def __init__(self, func):
        if not callable(func):
            raise TypeError('func must be callable')

        self.func = func

    def _call_func(self, xs):
        outs = self.func(*xs)

        if isinstance(outs, tuple):
            for i, out in enumerate(outs):
                if isinstance(out, variable.Variable):
                    continue
                n = i + 1
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(
                    n if n < 20 else n % 10, 'th')
                msg = ('{}{} element of a returned tuple is not Variable, '
                       'but is {}').format(n, suffix, type(out))
                raise RuntimeError(msg)
        elif isinstance(outs, variable.Variable):
            outs = (outs,)
        else:
            msg = ('A tuple of Variables or a Variable are expected, but {} '
                   'is returned.'.format(type(outs)))
            raise RuntimeError(msg)

        return outs

    def forward(self, inputs):
        xs = [variable.Variable(x, volatile=True) for x in inputs]
        outs = self._call_func(xs)
        return tuple(out.data for out in outs)

    def backward(self, inputs, grads):
        xs = [variable.Variable(x, volatile=False) for x in inputs]
        outs = self._call_func(xs)
        _DummyFunction(grads)(*outs).backward()
        return tuple(x.grad for x in xs)


def forget(func, *xs):
    """Call a function without storing internal results.

    On a forward propagation Chainer stores all internal results of
    :class:`Function` on a computational graph as they are required on
    backward-propagation. These results consume too much memory when the
    internal results are too large. This method **forgets** such internal
    results on forward propagation, and still supports back-propagation with
    recalculation.

    In a forward propagation, this method calls a given function with given
    variables without creating a computational graph. That means, no internal
    results are stored. In a backward propagation this method calls the given
    function again to create a computational graph to execute back-propagation.

    This method reduces internal memory usage. Instead it requires more
    calculation time as it calls the function twice.

    .. admonition:: Example

       Let ``f`` be a function defined as:

       >>> def f(a, b):
       ...   return a + b + a

       and, ``x`` and ``y`` be :class:`~chainer.Variable`:

       >>> x = chainer.Variable(np.random.uniform(-1, 1, 5).astype('f'))
       >>> y = chainer.Variable(np.random.uniform(-1, 1, 5).astype('f'))

       When ``z`` is calculated as ``z = f(x, y)``, its internal result
       ``x + y`` is stored in memory. Instead if you call ``f`` with
       :meth:`forget`:

       >>> z = F.forget(f, x, y)

       internal ``x + y`` is forgotten.

    .. note::

      The method does not support functions behaving randomly, such as
      :meth:`~chainer.functions.dropout` and
      :meth:`~chainer.functions.negative_sampling`. It is because first results
      of these function differ from the second one.

    Args:
        func (callable): A function to call. It needs to be called with
            :class:`~chainer.Variable` object(s) and to return a
            :class:`~chainer.Variable` object or a tuple of
            :class:`~chainer.Variable` objects.
        xs (~chainer.Variable): Argument variables of the function.

    Returns:
        ~chainer.Variable: A variable ``func`` returns. If it returns a tuple,
        the method returns a tuple too.

    """
    return Forget(func)(*xs)
