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
    return Forget(func)(*xs)
