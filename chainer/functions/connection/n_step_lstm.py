import itertools

import numpy
import six

from chainer import cuda
from chainer.functions.activation import lstm
from chainer.functions.array import concat
from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer.functions.array import stack
from chainer.functions.connection import linear
from chainer.functions.connection import n_step_rnn
from chainer.functions.connection.n_step_rnn import _stack_weight
from chainer.functions.connection.n_step_rnn import get_random_state
from chainer.functions.noise import dropout

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()


class NStepLSTM(n_step_rnn.BaseNStepRNN):
    def __init__(self, n_layers, states, train=True):
        n_step_rnn.BaseNStepRNN.__init__(self, n_layers, states,
                                         rnn_dir='uni', rnn_mode='lstm',
                                         train=train)


class NStepBiLSTM(n_step_rnn.BaseNStepRNN):
    def __init__(self, n_layers, states, train=True):
        n_step_rnn.BaseNStepRNN.__init__(self, n_layers, states,
                                         rnn_dir='bi', rnn_mode='lstm',
                                         train=train)


def n_step_lstm(
        n_layers, dropout_ratio, hx, cx, ws, bs, xs, train=True,
        use_cudnn=True):
    """Stacked Long Short-Term Memory function for sequence inputs.

    This function calculates stacked LSTM with sequences. This function gets
    an initial hidden state :math:`h_0`, an initial cell state :math:`c_0`,
    an input sequence :math:`x`, weight matrices :math:`W`, and bias vectors
    :math:`b`.
    This function calculates hidden states :math:`h_t` and :math:`c_t` for each
    time :math:`t` from input :math:`x_t`.

    .. math::
       i_t &= \\sigma(W_0 x_t + W_4 h_{t-1} + b_0 + b_4) \\\\
       f_t &= \\sigma(W_1 x_t + W_5 h_{t-1} + b_1 + b_5) \\\\
       o_t &= \\sigma(W_2 x_t + W_6 h_{t-1} + b_2 + b_6) \\\\
       a_t &= \\tanh(W_3 x_t + W_7 h_{t-1} + b_3 + b_7) \\\\
       c_t &= f_t \\dot c_{t-1} + i_t \\dot a_t \\\\
       h_t &= o_t \\dot \\tanh(c_t)

    As the function accepts a sequence, it calculates :math:`h_t` for all
    :math:`t` with one call. Eight weight matrices and eight bias vectors are
    required for each layers. So, when :math:`S` layers exists, you need to
    prepare :math:`8S` weigth matrices and :math:`8S` bias vectors.

    If the number of layers ``n_layers`` is greather than :math:`1`, input
    of ``k``-th layer is hidden state ``h_t`` of ``k-1``-th layer.
    Note that all input variables except first layer may have different shape
    from the first layer.

    Args:
        n_layers(int): Number of layers.
        dropout_ratio(float): Dropout ratio.
        hx (chainer.Variable): Variable holding stacked hidden states.
            Its shape is ``(S, B, N)`` where ``S`` is number of layers and is
            equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
            dimention of hidden units.
        cx (chainer.Variable): Variable holding stacked cell states.
            It has the same shape as ``hx``.
        ws (list of list of chainer.Variable): Weight matrices. ``ws[i]``
            represents weights for i-th layer.
            Each ``ws[i]`` is a list containing eight matrices.
            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
            Only ``ws[0][j]`` where ``0 <= j < 4`` is ``(I, N)`` shape as they
            are multiplied with input variables. All other matrices has
            ``(N, N)`` shape.
        bs (list of list of chainer.Variable): Bias vectors. ``bs[i]``
            represnents biases for i-th layer.
            Each ``bs[i]`` is a list containing eight vectors.
            ``bs[i][j]`` is corresponding with ``b_j`` in the equation.
            Shape of each matrix is ``(N,)`` where ``N`` is dimention of
            hidden units.
        xs (list of chainer.Variable): A list of :class:`~chainer.Variable`
            holding input values. Each element ``xs[t]`` holds input value
            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is
            mini-batch size for time ``t``, and ``I`` is size of input units.
            Note that this functions supports variable length sequences.
            When sequneces has different lengths, sort sequences in descending
            order by length, and transpose the sorted sequence.
            :func:`~chainer.functions.transpose_sequence` transpose a list
            of :func:`~chainer.Variable` holding sequence.
            So ``xs`` needs to satisfy
            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.
        train (bool): If ``True``, this function executes dropout.
        use_cudnn (bool): If ``True``, this function uses cuDNN if available.

    Returns:
        tuple: This functions returns a tuple concaining three elements,
            ``hy``, ``cy`` and ``ys``.
            - ``hy`` is an updated hidden states whose shape is same as ``hx``.
            - ``cy`` is an updated cell states whose shape is same as ``cx``.
            - ``ys`` is a list of :class:`~chainer.Variable` . Each element
              ``ys[t]`` holds hidden states of the last layer corresponding
              to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is
              mini-batch size for time ``t``, and ``N`` is size of hidden
              units. Note that ``B_t`` is the same value as ``xs[t]``.

    .. seealso::

       :func:`chainer.functions.lstm`

    """

    xp = cuda.get_array_module(hx, hx.data)

    if use_cudnn and xp is not numpy and cuda.cudnn_enabled and \
       _cudnn_version >= 5000:
        states = get_random_state().create_dropout_states(dropout_ratio)
        # flatten all input variables
        inputs = tuple(itertools.chain(
            (hx, cx),
            itertools.chain.from_iterable(ws),
            itertools.chain.from_iterable(bs),
            xs))
        rnn = NStepLSTM(n_layers, states, train=train)
        ret = rnn(*inputs)
        hy, cy = ret[:2]
        ys = ret[2:]
        return hy, cy, ys

    else:
        hx = split_axis.split_axis(hx, n_layers, axis=0, force_tuple=True)
        hx = [reshape.reshape(h, h.shape[1:]) for h in hx]
        cx = split_axis.split_axis(cx, n_layers, axis=0, force_tuple=True)
        cx = [reshape.reshape(c, c.shape[1:]) for c in cx]

        xws = [_stack_weight([w[2], w[0], w[1], w[3]]) for w in ws]
        hws = [_stack_weight([w[6], w[4], w[5], w[7]]) for w in ws]
        xbs = [_stack_weight([b[2], b[0], b[1], b[3]]) for b in bs]
        hbs = [_stack_weight([b[6], b[4], b[5], b[7]]) for b in bs]

        xs_next = xs
        hy = []
        cy = []
        for layer in six.moves.range(n_layers):
            h = hx[layer]
            c = cx[layer]
            h_forward = []
            c_forward = []
            for x in xs_next:
                batch = x.shape[0]
                if h.shape[0] > batch:
                    h, h_rest = split_axis.split_axis(h, [batch], axis=0)
                    c, c_rest = split_axis.split_axis(c, [batch], axis=0)
                else:
                    h_rest = None
                    c_rest = None

                x = dropout.dropout(x, ratio=dropout_ratio, train=train)
                h = dropout.dropout(h, ratio=dropout_ratio, train=train)
                lstm_in = linear.linear(x, xws[layer], xbs[layer]) + \
                    linear.linear(h, hws[layer], hbs[layer])

                c_bar, h_bar = lstm.lstm(c, lstm_in)
                if h_rest is not None:
                    h = concat.concat([h_bar, h_rest], axis=0)
                    c = concat.concat([c_bar, c_rest], axis=0)
                else:
                    h = h_bar
                    c = c_bar
                h_forward.append(h_bar)
                c_forward.append(c_bar)

            xs_next = h_forward

            hy.append(h)
            cy.append(c)

        ys = h_forward

    hy = stack.stack(hy)
    cy = stack.stack(cy)
    return hy, cy, tuple(ys)

def n_step_bilstm(
        n_layers, dropout_ratio, hx, cx, ws, bs, xs, train=True,
        use_cudnn=True):
    """Stacked Bi-direction Long Short-Term Memory function for sequence inputs.

    This function calculates stacked Bi-direction LSTM with sequences.
    This function gets an initial hidden state :math:`h_0`, an initial cell
    state :math:`c_0`, an input sequence :math:`x`, weight matrices :math:`W`,
    and bias vectors :math:`b`.
    This function calculates hidden states :math:`h_t` and :math:`c_t` for each
    time :math:`t` from input :math:`x_t`.

    .. math::
       i_t &= \\sigma(W_0 x_t + W_4 h_{t-1} + b_0 + b_4) \\\\
       f_t &= \\sigma(W_1 x_t + W_5 h_{t-1} + b_1 + b_5) \\\\
       o_t &= \\sigma(W_2 x_t + W_6 h_{t-1} + b_2 + b_6) \\\\
       a_t &= \\tanh(W_3 x_t + W_7 h_{t-1} + b_3 + b_7) \\\\
       c_t &= f_t \\dot c_{t-1} + i_t \\dot a_t \\\\
       h_t &= o_t \\dot \\tanh(c_t)

    As the function accepts a sequence, it calculates :math:`h_t` for all
    :math:`t` with one call. Eight weight matrices and eight bias vectors are
    required for each layers. So, when :math:`S` layers exists, you need to
    prepare :math:`8S` weigth matrices and :math:`8S` bias vectors.

    If the number of layers ``n_layers`` is greather than :math:`1`, input
    of ``k``-th layer is hidden state ``h_t`` of ``k-1``-th layer.
    Note that all input variables except first layer may have different shape
    from the first layer.

    Args:
        n_layers(int): Number of layers.
        dropout_ratio(float): Dropout ratio.
        hx (chainer.Variable): Variable holding stacked hidden states.
            Its shape is ``(S, B, N)`` where ``S`` is number of layers and is
            equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
            dimention of hidden units.
        cx (chainer.Variable): Variable holding stacked cell states.
            It has the same shape as ``hx``.
        ws (list of list of chainer.Variable): Weight matrices. ``ws[i]``
            represents weights for i-th layer.
            Each ``ws[i]`` is a list containing eight matrices.
            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
            Only ``ws[0][j]`` where ``0 <= j < 4`` is ``(I, N)`` shape as they
            are multiplied with input variables. All other matrices has
            ``(N, N)`` shape.
        bs (list of list of chainer.Variable): Bias vectors. ``bs[i]``
            represnents biases for i-th layer.
            Each ``bs[i]`` is a list containing eight vectors.
            ``bs[i][j]`` is corresponding with ``b_j`` in the equation.
            Shape of each matrix is ``(N,)`` where ``N`` is dimention of
            hidden units.
        xs (list of chainer.Variable): A list of :class:`~chainer.Variable`
            holding input values. Each element ``xs[t]`` holds input value
            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is
            mini-batch size for time ``t``, and ``I`` is size of input units.
            Note that this functions supports variable length sequences.
            When sequneces has different lengths, sort sequences in descending
            order by length, and transpose the sorted sequence.
            :func:`~chainer.functions.transpose_sequence` transpose a list
            of :func:`~chainer.Variable` holding sequence.
            So ``xs`` needs to satisfy
            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.
        train (bool): If ``True``, this function executes dropout.
        use_cudnn (bool): If ``True``, this function uses cuDNN if available.

    Returns:
        tuple: This functions returns a tuple concaining three elements,
            ``hy``, ``cy`` and ``ys``.
            - ``hy`` is an updated hidden states whose shape is same as ``hx``.
            - ``cy`` is an updated cell states whose shape is same as ``cx``.
            - ``ys`` is a list of :class:`~chainer.Variable` . Each element
              ``ys[t]`` holds hidden states of the last layer corresponding
              to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is
              mini-batch size for time ``t``, and ``N`` is size of hidden
              units. Note that ``B_t`` is the same value as ``xs[t]``.

    .. seealso::

       :func:`chainer.functions.lstm`

    """

    xp = cuda.get_array_module(hx, hx.data)

    if use_cudnn and xp is not numpy and cuda.cudnn_enabled and \
       _cudnn_version >= 5000:
        states = get_random_state().create_dropout_states(dropout_ratio)
        # flatten all input variables
        inputs = tuple(itertools.chain(
            (hx, cx),
            itertools.chain.from_iterable(ws),
            itertools.chain.from_iterable(bs),
            xs))
        rnn = NStepBiLSTM(n_layers, states, train=train)
        ret = rnn(*inputs)
        hy, cy = ret[:2]
        ys = ret[2:]
        return hy, cy, ys

    else:
        hx = split_axis.split_axis(hx, n_layers * 2, axis=0, force_tuple=True)
        hx = [reshape.reshape(h, h.shape[1:]) for h in hx]
        cx = split_axis.split_axis(cx, n_layers * 2, axis=0, force_tuple=True)
        cx = [reshape.reshape(c, c.shape[1:]) for c in cx]

        xws = [_stack_weight([w[2], w[0], w[1], w[3]]) for w in ws]
        hws = [_stack_weight([w[6], w[4], w[5], w[7]]) for w in ws]
        xbs = [_stack_weight([b[2], b[0], b[1], b[3]]) for b in bs]
        hbs = [_stack_weight([b[6], b[4], b[5], b[7]]) for b in bs]

        xs_next = xs
        hy = []
        cy = []
        for layer in six.moves.range(n_layers):
            def _one_directional_loop(di):
                h_list = []
                c_list = []
                layer_idx = 2 * layer + di
                h = hx[layer_idx]
                c = cx[layer_idx]
                if di == 0:
                    xs_list = xs_next
                else:
                    xs_list = reversed(xs_next)
                for x in xs_list:
                    batch = x.shape[0]
                    if h.shape[0] > batch:
                        h, h_rest = split_axis.split_axis(h, [batch], axis=0)
                        c, c_rest = split_axis.split_axis(c, [batch], axis=0)
                    else:
                        h_rest = None
                        c_rest = None
                    x = dropout.dropout(x, ratio=dropout_ratio, train=train)
                    lstm_in = linear.linear(x, xws[layer_idx], xbs[layer_idx]) + \
                        linear.linear(h, hws[layer_idx], hbs[layer_idx])

                    c_bar, h_bar = lstm.lstm(c, lstm_in)
                    if h_rest is not None:
                        h = concat.concat([h_bar, h_rest], axis=0)
                        c = concat.concat([c_bar, c_rest], axis=0)
                    else:
                        h = h_bar
                        c = c_bar
                    h_list.append(h_bar)
                    c_list.append(c_bar)
                return h, c, h_list, c_list

            h, c, h_forward, c_forward = _one_directional_loop(di=0)
            hy.append(h)
            cy.append(c)

            h, c, h_backward, c_backward = _one_directional_loop(di=1)
            hy.append(h)
            cy.append(c)

            h_backward.reverse()
            # concat
            xs_next = [concat.concat([hfi, hbi], axis=1) for (hfi, hbi) in
                       zip(h_forward, h_backward)]

        # last layer
        ys = [concat.concat([hfi, hbi], axis=1) for (hfi, hbi) in
              zip(h_forward, h_backward)]

        hy = stack.stack(hy)
        cy = stack.stack(cy)
        return hy, cy, tuple(ys)
