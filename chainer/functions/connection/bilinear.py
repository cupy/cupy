import math

import numpy

from chainer import cuda
from chainer import function
from chainer.utils import array
from chainer.utils import type_check


class Bilinear(function.Function):

    """Bilinear function, an extension of Linear function.

    ``Bilinear`` function takes two input vectors and outputs one vector.
    If one of the input vectors is fixed, this function works
    as an affine transform of the other input vector.

    ``Bilinear`` function is a building block of Neural Tensor Network
    (See the reference paper below).

    To be precise, ``Bilinear`` function has four parameters,
    :math:`W\in \mathbb{R}^{J \cdot K \cdot L}`,
    :math:`V^1\in \mathbb{R}^{J \cdot L}`,
    :math:`V^2\in \mathbb{R}^{K \cdot L}`, and :math:`b\in \mathbb{R}^{L}`.
    In this document, we call :math:`V^1`, :math:`V^2`,
    and :math:`b` linear parameters.

    Given two inputs (in a mini-batch manner)
    :math:`e^1\in \mathbb{R}^{I\cdot J}` and
    :math:`e^2\in \mathbb{R}^{I\cdot K}`
    where :math:`I` is mini-batch size, the output of forward propagation is
    calculated as

    .. math::

      y_{il} = \sum_{jk} e^1_{ij} e^2_{ik} W_{jkl} + \
        \sum_{j} e^1_{ij} V^1_{jl} + \sum_{k} e^2_{ik} V^2_{kl} + b_{l}.

    If ``nobias`` option is set ``True``, ``Bilinear`` function does
    not have linear parameters, that is, the last three term is omitted
    and only :math:`W` works as the parameter.

    .. note::

       ``Bilinear`` function accepts an input variable of a non-matrix array.
       In this case, the leading dimension is treated as the batch dimension,
       and the other dimensions are reduced to one dimension.

    .. note::

       In the original paper, :math:`J` and :math:`K`
       must be equal and the author denotes :math:`[V^1 V^2]`
       (concatenation of matrices) by :math:`V`.

    Args:
        left_size (int): Dimension of input vector :math:`e^1` (:math:`J`)
        right_size (int): Dimension of input vector :math:`e^2` (:math:`K`)
        out_size (int): Dimension of output vector :math:`y` (:math:`L`)
        nobias (bool): If ``True``, linear parameters are omitted.
        initialW (3-D Array): Initial value of :math:`W`.
            Shape of this argument must be
            ``(left_size, right_size, out_size)``. If ``None``,
            :math:`W` is initialized by centered Gaussian distribution properly
            scaled according to the dimension of inputs and outputs.
        initial_bias (tuple): Intial values of :math:`V^1`, :math:`V^2`
            and :math:`b`. The length this argument must be 3.
            Each element of this tuple must have the shapes of
            ``(left_size, output_size)``, ``(right_size, output_size)``,
            and ``(output_size,)``, respectively. If ``None``, :math:`V^1`
            and :math:`V^2` is initialized by scaled centered Gaussian
            distributions and :math:`b` is set to :math:`0`.

    See:
        `Reasoning With Neural Tensor Networks for Knowledge Base Completion
        <http://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-
        networks-for-knowledge-base-completion>`_ [Socher+, NIPS2013].
    """

    def __init__(self, left_size, right_size, out_size, nobias=False,
                 initialW=None, initial_bias=None):

        self.W = None
        self.gW = None
        self.V1 = None
        self.gV1 = None
        self.V2 = None
        self.gV2 = None
        self.b = None
        self.gb = None

        self.in_sizes = (left_size, right_size)
        self.nobias = nobias

        if initialW is not None:
            assert initialW.shape == (
                self.in_sizes[0], self.in_sizes[1], out_size)
            self.W = initialW
        else:
            # TODO(Kenta OONO): I do not know appropriate way of
            # initializing weights in tensor network.
            # This initialization is a modification of
            # that of Linear function.
            in_size = numpy.prod(self.in_sizes)
            self.W = numpy.random.normal(
                0, math.sqrt(1. / in_size),
                (self.in_sizes[0], self.in_sizes[1], out_size)
            ).astype(numpy.float32)

        if not self.nobias:
            if initial_bias is not None:
                assert len(initial_bias) == 3
                assert initial_bias[0].shape == (self.in_sizes[0], out_size)
                assert initial_bias[1].shape == (self.in_sizes[1], out_size)
                assert initial_bias[2].shape == (out_size,)
                self.V1, self.V2, self.b = initial_bias
            else:
                self.V1 = numpy.random.normal(
                    0, math.sqrt(1. / self.in_sizes[0]),
                    (self.in_sizes[0], out_size)).astype(numpy.float32)
                self.V2 = numpy.random.normal(
                    0, math.sqrt(1. / self.in_sizes[1]),
                    (self.in_sizes[1], out_size)).astype(numpy.float32)
                self.b = numpy.zeros((out_size, ), dtype=numpy.float32)

        xp = cuda.get_array_module(self.W)
        self.gW = xp.full_like(self.W, numpy.nan)
        if not self.nobias:
            self.gV1 = xp.full_like(self.V1, numpy.nan)
            self.gV2 = xp.full_like(self.V2, numpy.nan)
            self.gb = xp.full_like(self.b, numpy.nan)

    @property
    def parameter_names(self):
        if self.nobias:
            return 'W',
        assert self.V1 is not None
        assert self.V2 is not None
        assert self.b is not None
        return 'W', 'V1', 'V2', 'b'

    @property
    def gradient_names(self):
        if self.nobias:
            return 'gW',
        assert self.gV1 is not None
        assert self.gV2 is not None
        assert self.gb is not None
        return 'gW', 'gV1', 'gV2', 'gb'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        e1_type, e2_type = in_types
        type_check.expect(
            e1_type.ndim >= 2,
            e2_type.ndim >= 2,
            e1_type.shape[0] == e2_type.shape[0]
        )

        in_sizes = type_check.Variable(self.in_sizes, 'in_sizes')
        type_check_prod = type_check.Variable(numpy.prod, 'prod')
        type_check.expect(
            type_check_prod(e1_type.shape[1:]) == in_sizes[0],
            type_check_prod(e2_type.shape[1:]) == in_sizes[1]
        )

    def zero_grads(self):
        self.gW.fill(0)
        if not self.nobias:
            self.gV1.fill(0)
            self.gV2.fill(0)
            self.gb.fill(0)

    def forward_cpu(self, x):
        e1 = array.as_mat(x[0])
        e2 = array.as_mat(x[1])
        y = numpy.einsum('ij,ik,jkl->il', e1, e2, self.W)
        if not self.nobias:
            y += e1.dot(self.V1)
            y += e2.dot(self.V2)
            y += self.b
        return y,

    def forward_gpu(self, x):
        e1 = array.as_mat(x[0])
        e2 = array.as_mat(x[1])
        i_len, j_len = e1.shape
        k_len = e2.shape[1]

        # 'ij,ik->ijk'
        e1e2 = e1[:, :, numpy.newaxis] * e2[:, numpy.newaxis, :]

        # ijk->i[jk]
        e1e2 = e1e2.reshape(i_len, j_len * k_len)
        # jkl->[jk]l
        W_mat = self.W.reshape(
            self.W.shape[0] * self.W.shape[1], self.W.shape[2])

        # 'i[jk],[jk]l->il'
        y = e1e2.dot(W_mat)

        if not self.nobias:
            y += e1.dot(self.V1)
            y += e2.dot(self.V2)
            y += self.b
        return y,

    def backward_cpu(self, x, gy):
        e1 = array.as_mat(x[0])
        e2 = array.as_mat(x[1])
        gy, = gy

        self.gW += numpy.einsum('ij,ik,il->jkl', e1, e2, gy)
        if not self.nobias:
            self.gV1 += e1.T.dot(gy)
            self.gV2 += e2.T.dot(gy)
            self.gb += gy.sum(0)

        ge1 = numpy.einsum('ik,jkl,il->ij', e2, self.W, gy)
        ge2 = numpy.einsum('ij,jkl,il->ik', e1, self.W, gy)
        if not self.nobias:
            ge1 += gy.dot(self.V1.T)
            ge2 += gy.dot(self.V2.T)
        return (ge1.reshape(x[0].shape), ge2.reshape(x[1].shape))

    def backward_gpu(self, x, gy):
        e1 = array.as_mat(x[0])
        e2 = array.as_mat(x[1])
        gy, = gy

        kern_add = cuda.reduce(
            'T in0, T in1, T in2', 'T out',
            'in0 * in1 * in2', 'a + b', 'out += a', 0,
            'bilinear_product_add')
        kern = cuda.reduce(
            'T in0, T in1, T in2', 'T out',
            'in0 * in1 * in2', 'a + b', 'out = a', 0,
            'bilinear_product')

        e1_b = e1[:, :, numpy.newaxis, numpy.newaxis]  # ij
        e2_b = e2[:, numpy.newaxis, :, numpy.newaxis]  # ik
        gy_b = gy[:, numpy.newaxis, numpy.newaxis, :]  # il
        W_b = self.W[numpy.newaxis, :, :, :]  # jkl

        # 'ij,ik,il->jkl'
        kern_add(e1_b, e2_b, gy_b, self.gW, axis=0)

        if not self.nobias:
            self.gV1 += e1.T.dot(gy)
            self.gV2 += e2.T.dot(gy)
            self.gb += gy.sum(axis=0)

        # 'ik,jkl,il->ij'
        ge1 = kern(e2_b, W_b, gy_b, axis=(2, 3))
        # 'ij,jkl,il->ik'
        ge2 = kern(e1_b, W_b, gy_b, axis=(1, 3))

        if not self.nobias:
            ge1 += gy.dot(self.V1.T)
            ge2 += gy.dot(self.V2.T)
        return (ge1.reshape(x[0].shape), ge2.reshape(x[1].shape))
