from chainer import cuda
from chainer.functions.activation import relu
from chainer.functions.activation import sigmoid
from chainer.functions.math import basic_math
from chainer import link
from chainer.links.connection import linear
import copy


class Highway(link.Chain):

    """Highway module.

    Network allowing unimpeded information flow across layer on information
    flow.
    It applies two different gate. One gate is transform gate, and the other
    is carry gate. Carry gate enable to propagete large gradient value to
    previous layer.

    The output is sum of linear multiplied transform gate, and input array
    multiplied carry gate.
    Its array has the same spatial size as the input. In order to satisfy this,
    Highway module uses square matrix as its weight.

    See: `Highway Networks <https://arxiv.org/abs/1505.00387>`_.

    Args:
        in_out_size (int): Dimension of input and output vectors.
        nobias (bool): If ``True``, then this function does not use the bias.
        activate: Activation function of plain array. tanh is also enable.
        init_Wh (2-D array): Initial weight value of plain array. If ``None``,
            then this function uses to initialize ``wscale``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        init_bh (1-D array): Initial bias value of plain array. If ``None``,
            then this function uses to initialize zero vector.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        init_Wt (2-D array): Initial weight value of transform array.
            If ``None``, then this function uses to initialize ``wscale``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        init_bt (1-D array): Initial bias value of transform array.
            Default value is -1 vector.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            Negative value is sufficient for learning (e.g. -1, -3, ...).
    """

    def __init__(self, in_out_size, nobias=False, activate=relu.relu,
                 init_Wh=None, init_Wt=None, init_bh=None, init_bt=-1):
        super(Highway, self).__init__(
            plain=linear.Linear(in_out_size, in_out_size, nobias=nobias,
                                initialW=init_Wh, initial_bias=init_bh),
            transform=linear.Linear(in_out_size, in_out_size, nobias=nobias,
                                    initialW=init_Wt, initial_bias=init_bt)
        )
        self.activate = activate

    def __call__(self, x):
        """Computes the output of the Highway module.

        Args:
            x (~chainer.Variable): Input variable.

        Returns:
            Variable: Output variable. Its array has the same spatial size and
            the same minibatch size as the input array.

        """
        out_plain = self.activate(self.plain(x))
        out_transform = sigmoid.sigmoid(self.transform(x))
        out_transform_copy = copy.copy(out_transform)
        a = basic_math.mul(out_plain, out_transform)
        ones = cuda.get_array_module(x).ones_like(x.data)
        b = basic_math.rsub(out_transform_copy, ones)
        c = basic_math.mul(x, b)
        y = basic_math.add(a, c)
        return y
