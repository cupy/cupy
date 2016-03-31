from chainer.functions.array import broadcast
from chainer.functions.array import reshape
from chainer.functions.array import select_item
from chainer.functions.connection import embed_id
from chainer.functions.math import logsumexp
from chainer.functions.math import minmax
from chainer.functions.math import sum as _sum


def crf1d(cost, xs, ys):
    assert xs[0].data.shape[1] == cost.data.shape[0]

    n_label = cost.data.shape[0]
    n_batch = xs[0].data.shape[0]

    alpha = xs[0]
    for x in xs[1:]:
        b_alpha, b_cost = broadcast.broadcast(
            reshape.reshape(alpha, (n_batch, n_label, 1)), cost)
        alpha = logsumexp.logsumexp(b_alpha + b_cost, axis=1) + x

    logz = logsumexp.logsumexp(alpha, axis=1)

    score = 0
    cost = reshape.reshape(cost, (cost.data.size, 1))
    for y1, y2 in zip(ys[:-1], ys[1:]):
        score += reshape.reshape(
            embed_id.embed_id(y1 * n_label + y2, cost), (n_batch,))
    for x, y in zip(xs, ys):
        score += select_item.select_item(x, y)

    return _sum.sum(logz - score) / n_batch


def crf1d_viterbi(cost, xs):
    n_label = cost.data.shape[0]
    n_batch = xs[0].data.shape[0]

    alpha = xs[0]
    max_inds = []
    for x in xs[1:]:
        b_alpha, b_cost = broadcast.broadcast(
            reshape.reshape(alpha, (n_batch, n_label, 1)), cost)
        scores = b_alpha + b_cost
        max_ind = minmax.argmax(scores, axis=1)
        max_inds.append(max_ind)
        alpha = minmax.max(scores, axis=1) + x

    inds = minmax.argmax(alpha, axis=1).data
    path = [inds]
    for m in reversed(max_inds):
        inds = m.data[range(len(inds)), inds]
        path.append(inds)
    path.reverse()

    return alpha, path
