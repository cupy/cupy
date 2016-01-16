import numpy

from chainer import cuda
from chainer import function
from chainer.utils import array
from chainer.utils import type_check


class BilinearFunction(function.Function):

    def check_type_forward(self, in_types):
        n_in = in_types.size().eval()
        if n_in != 3 and n_in != 6:
            raise type_check.InvalidType(
                '%s or %s' % (in_types.size() == 3, in_types.size() == 6),
                '%s == %s' % (in_types.size(), n_in))

        e1_type, e2_type, W_type = in_types[:3]
        type_check_prod = type_check.Variable(numpy.prod, 'prod')
        type_check.expect(
            e1_type.dtype == numpy.float32,
            e1_type.ndim >= 2,
            e2_type.dtype == numpy.float32,
            e2_type.ndim >= 2,
            e1_type.shape[0] == e2_type.shape[0],
            W_type.dtype == numpy.float32,
            W_type.ndim == 3,
            type_check_prod(e1_type.shape[1:]) == W_type.shape[0],
            type_check_prod(e2_type.shape[1:]) == W_type.shape[1],
        )

        if n_in == 6:
            out_size = W_type.shape[2]
            V1_type, V2_type, b_type = in_types[3:]
            type_check.expect(
                V1_type.dtype == numpy.float32,
                V1_type.ndim == 2,
                V1_type.shape[0] == W_type.shape[0],
                V1_type.shape[1] == out_size,
                V2_type.dtype == numpy.float32,
                V2_type.ndim == 2,
                V2_type.shape[0] == W_type.shape[1],
                V2_type.shape[1] == out_size,
                b_type.dtype == numpy.float32,
                b_type.ndim == 1,
                b_type.shape[0] == out_size,
            )

    def forward(self, inputs):
        e1 = array.as_mat(inputs[0])
        e2 = array.as_mat(inputs[1])
        W = inputs[2]

        xp = cuda.get_array_module(*inputs)
        if xp is numpy:
            y = numpy.einsum('ij,ik,jkl->il', e1, e2, W)
        else:
            i_len, j_len = e1.shape
            k_len = e2.shape[1]
            # 'ij,ik->ijk'
            e1e2 = e1[:, :, None] * e2[:, None, :]
            # ijk->i[jk]
            e1e2 = e1e2.reshape(i_len, j_len * k_len)
            # jkl->[jk]l
            W_mat = W.reshape(-1, W.shape[2])
            # 'i[jk],[jk]l->il'
            y = e1e2.dot(W_mat)

        if len(inputs) == 6:
            V1, V2, b = inputs[3:]
            y += e1.dot(V1)
            y += e2.dot(V2)
            y += b
        return y,

    def backward(self, inputs, grad_outputs):
        e1 = array.as_mat(inputs[0])
        e2 = array.as_mat(inputs[1])
        W = inputs[2]
        gy = grad_outputs[0]

        xp = cuda.get_array_module(*inputs)
        if xp is numpy:
            gW = numpy.einsum('ij,ik,il->jkl', e1, e2, gy)
            ge1 = numpy.einsum('ik,jkl,il->ij', e2, W, gy)
            ge2 = numpy.einsum('ij,jkl,il->ik', e1, W, gy)
        else:
            kern = cuda.reduce('T in0, T in1, T in2', 'T out',
                               'in0 * in1 * in2', 'a + b', 'out = a', 0,
                               'bilinear_product')

            e1_b = e1[:, :, None, None]  # ij
            e2_b = e2[:, None, :, None]  # ik
            gy_b = gy[:, None, None, :]  # il
            W_b = W[None, :, :, :]  # jkl

            gW = kern(e1_b, e2_b, gy_b, axis=0)  # 'ij,ik,il->jkl'
            ge1 = kern(e2_b, W_b, gy_b, axis=(2, 3))  # 'ik,jkl,il->ij'
            ge2 = kern(e1_b, W_b, gy_b, axis=(1, 3))  # 'ij,jkl,il->ik'

        ret = ge1.reshape(inputs[0].shape), ge2.reshape(inputs[1].shape), gW
        if len(inputs) == 6:
            V1, V2, b = inputs[3:]
            gV1 = e1.T.dot(gy)
            gV2 = e2.T.dot(gy)
            gb = gy.sum(0)
            ge1 += gy.dot(V1.T)
            ge2 += gy.dot(V2.T)
            ret += gV1, gV2, gb
        return ret


def bilinear(e1, e2, W, V1=None, V2=None, b=None):
    """Applies a bilinear function based on given parameters.

    This is a building block of Neural Tensor Network (see the reference paper
    below). It takes two input variables and one or four parameters, and
    outputs one variable.

    To be precise, denote six input arrays mathematically by
    :math:`e^1\\in \\mathbb{R}^{I\\cdot J}`,
    :math:`e^2\\in \\mathbb{R}^{I\\cdot K}`,
    :math:`W\\in \\mathbb{R}^{J \\cdot K \\cdot L}`,
    :math:`V^1\\in \\mathbb{R}^{J \\cdot L}`,
    :math:`V^2\\in \\mathbb{R}^{K \\cdot L}`, and
    :math:`b\\in \\mathbb{R}^{L}`,
    where :math:`I` is mini-batch size.
    In this document, we call :math:`V^1`, :math:`V^2`, and :math:`b` linear
    parameters.

    The output of forward propagation is calculated as

    .. math::

      y_{il} = \\sum_{jk} e^1_{ij} e^2_{ik} W_{jkl} + \\
        \\sum_{j} e^1_{ij} V^1_{jl} + \\sum_{k} e^2_{ik} V^2_{kl} + b_{l}.

    Note that V1, V2, b are optional. If these are not given, then this
    function omits the last three terms in the above equation.

    .. note::

       This function accepts an input variable ``e1`` or ``e2`` of a non-matrix
       array. In this case, the leading dimension is treated as the batch
       dimension, and the other dimensions are reduced to one dimension.

    .. note::

       In the original paper, :math:`J` and :math:`K`
       must be equal and the author denotes :math:`[V^1 V^2]`
       (concatenation of matrices) by :math:`V`.

    Args:
        e1 (~chainer.Variable): Left input variable.
        e2 (~chainer.Variable): Right input variable.
        W (~chainer.Variable): Quadratic weight variable.
        V1 (~chainer.Variable): Left coefficient variable.
        V2 (~chainer.Variable): Right coefficient variable.
        b (~chainer.Variable): Bias variable.

    Returns:
        ~chainer.Variable: Output variable.

    See:
        `Reasoning With Neural Tensor Networks for Knowledge Base Completion
        <http://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-
        networks-for-knowledge-base-completion>`_ [Socher+, NIPS2013].

    """
    flags = [V1 is None, V2 is None, b is None]
    if any(flags):
        if not all(flags):
            raise ValueError('All coefficients and bias for bilinear() must '
                             'be None, if at least one of them is None.')
        return BilinearFunction()(e1, e2, W)
    else:
        return BilinearFunction()(e1, e2, W, V1, V2, b)
