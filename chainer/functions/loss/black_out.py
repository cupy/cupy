from chainer.functions.array import broadcast
from chainer.functions.array import concat
from chainer.functions.array import expand_dims
from chainer.functions.array import reshape
from chainer.functions.connection import embed_id
from chainer.functions.math import exponential
from chainer.functions.math import logsumexp
from chainer.functions.math import matmul
from chainer.functions.math import sum as _sum


def black_out(x, t, W, samples, reduce='mean'):
    """BlackOut loss function.

    BlackOut loss function is defined as

    .. math::

      -\\log(p(t)) - \\sum_{s \\in S} \\log(1 - p(s)),

    where :math:`t` is the correct label, :math:`S` is a set of negative
    examples and :math:`p(\cdot)` is likelihood of a given label.
    And, :math:`p` is defined as

    .. math::

       p(y) = \\frac{\\exp(W_y^\\top x)}{
       \\sum_{s \\in samples} \\exp(W_s^\\top x)}.

    The output is a variable whose value depends on the value of
    the option ``reduce``. If it is ``'no'``, it holds the
    no loss values. If it is ``'mean'``, this function takes
    a mean of loss values.

    Args:
        x (~chainer.Variable): Batch of input vectors.
            Its shape should be :math:`(N, D)`.
        t (~chainer.Variable): Vector of ground truth labels.
            Its shape should be :math:`(N,)`. Each elements :math:`v`
            should satisfy :math:`0 \geq v \geq V` or :math:`-1`
            where :math:`V` is the number of label types.
        W (~chainer.Variable): Weight matrix.
            Its shape should be :math:`(V, D)`
        samples (~chainer.Variable): Negative samples.
            Its shape should be :math:`(N, S)` where :math:`S` is
            the number of negative samples.
        recude (str): Reduction option. Its value must be either
        ``'no'`` or ``'mean'``. Otherwise,
        :class:`ValueError` is raised.

    Returns:
        ~chainer.Variable:
            A variable object holding loss value(s).
            If ``reduce`` is ``'no'``, the output variable holds an
            array whose shape is :math:`(N,)` .
            If it is ``'mean'``, it holds a scalar.

    See: `BlackOut: Speeding up Recurrent Neural Network Language Models With \
         Very Large Vocabularies <https://arxiv.org/abs/1511.06909>`_

    .. seealso:: :class:`~chainer.links.BlackOut`.

    """

    batch_size = x.shape[0]

    neg_emb = embed_id.embed_id(samples, W)
    neg_y = matmul.batch_matmul(neg_emb, x)
    neg_y = reshape.reshape(neg_y, neg_y.shape[:-1])

    pos_emb = expand_dims.expand_dims(embed_id.embed_id(t, W), 1)
    pos_y = matmul.batch_matmul(pos_emb, x)
    pos_y = reshape.reshape(pos_y, pos_y.shape[:-1])

    logz = logsumexp.logsumexp(concat.concat([pos_y, neg_y]), axis=1)
    blogz, bneg_y = broadcast.broadcast(
        reshape.reshape(logz, (batch_size, 1)), neg_y)
    ny = exponential.log(1 - exponential.exp(bneg_y - blogz))
    py = reshape.reshape(pos_y, (batch_size,))
    loss = -(py - logz + _sum.sum(ny, axis=1))
    if reduce == 'mean':
        return _sum.sum(loss) / batch_size
    else:
        return loss
