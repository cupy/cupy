import collections

try:
    import theano
    _available = True
except ImportError:
    _available = False

from chainer.functions.theano import theano_function
from chainer import link
from chainer import utils


def _to_var_tuple(vs):
    msg = ('inputs and outputs must be a TensorVariable, a list '
           'of TensorVariable or a tuple of TensorVariable')

    if isinstance(vs, theano.tensor.TensorVariable):
        return vs,
    elif isinstance(vs, collections.Iterable):
        vs = tuple(vs)
        if not all(isinstance(v, theano.tensor.TensorVariable) for v in vs):
            raise TypeError(msg)
        return vs
    else:
        raise TypeError(msg)


class TheanoFunction(link.Link):

    """Theano function wrapper.

    This function wrapps Theano function as a :class:`chainer.Link`.
    A user needs to make input Theano variables and output Theano variables.
    This function automatically creates Theano function for forward calculation
    and backward calculation from inputs and ouptuts. And then, it sends data
    in :class:`chainer.Variable` to the function and gets results from Theano.

    .. admonition:: Example

       >>> import theano
       >>> x = theano.tensor.fvector()
       >>> y = theano.tensor.fvector()
       >>> z = x + y
       >>> w = x - y
       >>> f = L.TheanoFunction(inputs=[x, y], outputs=[z, w])
       >>> a = chainer.Variable(numpy.array([1, 2], dtype='f'))
       >>> b = chainer.Variable(numpy.array([2, 3], dtype='f'))
       >>> c, d = f(a, b)
       >>> c.data
       array([ 3.,  5.], dtype=float32)
       >>> d.data
       array([-1., -1.], dtype=float32)

    .. note::

       The current implementation always copys :class:`cupy.ndarray` to CPU.

    Args:
        inputs (tuple of ~theano.tensor.TensorVariable): Input variables of
            Theano. This function accepts the same number of
            :class:`~chainer.Variable`s in forward computation.
        outputs (tuple of ~theano.tensor.TensorVariable): Output variables of
            Theano. The function returns the same number
            :class:`~chainder.Variable`s as ``outputs``.

    """

    def __init__(self, inputs, outputs):
        utils.experimental('chainer.links.TheanoFunction')
        if not _available:
            msg = '''theano is not installed on your environment.
Please install theano to activate theano function.

  $ pip install theano'''
            raise RuntimeError(msg)

        inputs = _to_var_tuple(inputs)
        outputs = _to_var_tuple(outputs)

        # TODO(unno): We can remove redundant gpu-cpu copy using
        # theano.sandbox.cuda.basic_ops.gpu_from_host
        self.forward_func = theano.function(inputs=inputs, outputs=outputs)

        gs = tuple(
            o.type('g_{}'.format(i)) for i, o in enumerate(outputs))
        known_grads = dict(zip(outputs, gs))
        grad = theano.tensor.grad(
            cost=None, wrt=inputs, known_grads=known_grads,
            disconnected_inputs='ignore')

        self.backward_func = theano.function(
            inputs=inputs + gs,
            outputs=grad,
            on_unused_input='ignore')

    def __call__(self, *args):
        return theano_function.theano_function(
            self.forward_func, self.backward_func, *args)
