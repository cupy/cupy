import itertools

import numpy
import six

from chainer import cuda
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.array import concat
from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer.functions.array import stack
from chainer.functions.connection import linear
from chainer.functions.connection import n_step_rnn
from chainer.functions.connection.n_step_rnn import get_random_state
from chainer.functions.noise import dropout

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()


class NStepGRU(n_step_rnn.BaseNStepRNN):

    def __init__(self, n_layers, states, train=True):
        n_step_rnn.BaseNStepRNN.__init__(self, n_layers, states,
                                         rnn_dir='uni', rnn_mode='gru',
                                         train=train)


class NStepBiGRU(n_step_rnn.BaseNStepRNN):

    def __init__(self, n_layers, states, train=True):
        n_step_rnn.BaseNStepRNN.__init__(self, n_layers, states,
                                         rnn_dir='bi', rnn_mode='gru',
                                         train=train)


def n_step_gru(
        n_layers, dropout_ratio, hx, ws, bs, xs, train=True,
        use_cudnn=True):
    """Stacked Uni-directional Gated Recurrent Unit function.

    This function calculates stacked Uni-directional GRU with sequences.
    This function gets an initial hidden state :math:`h_0`, an input
    sequence :math:`x`, weight matrices :math:`W`, and bias vectors :math:`b`.
    This function calculates hidden states :math:`h_t` for each time :math:`t`
    from input :math:`x_t`.

    .. math::
       r_t &= \\sigma(W_0 x_t + W_3 h_{t-1} + b_0 + b_3) \\\\
       z_t &= \\sigma(W_1 x_t + W_4 h_{t-1} + b_1 + b_4) \\\\
       h'_t &= \\tanh(W_2 x_t + b_2 + r_t \\cdot (W_5 h_{t-1} + b_5)) \\\\
       h_t &= (1 - z_t) \\cdot h'_t + z_t \\cdot h_{t-1}

    As the function accepts a sequence, it calculates :math:`h_t` for all
    :math:`t` with one call. Six weight matrices and six bias vectors are
    required for each layers. So, when :math:`S` layers exists, you need to
    prepare :math:`6S` weigth matrices and :math:`6S` bias vectors.

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
        ws (list of list of chainer.Variable): Weight matrices. ``ws[i]``
            represents weights for i-th layer.
            Each ``ws[i]`` is a list containing six matrices.
            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
            Only ``ws[0][j]`` where ``0 <= j < 3`` is ``(I, N)`` shape as they
            are multiplied with input variables. All other matrices has
            ``(N, N)`` shape.
        bs (list of list of chainer.Variable): Bias vectors. ``bs[i]``
            represnents biases for i-th layer.
            Each ``bs[i]`` is a list containing six vectors.
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
            ``hy`` and ``ys``.
            - ``hy`` is an updated hidden states whose shape is same as ``hx``.
            - ``ys`` is a list of :class:`~chainer.Variable` . Each element
              ``ys[t]`` holds hidden states of the last layer corresponding
              to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is
              mini-batch size for time ``t``, and ``N`` is size of hidden
              units. Note that ``B_t`` is the same value as ``xs[t]``.

    """

    return n_step_gru_base(n_layers, dropout_ratio, hx, ws, bs, xs, train,
                           use_cudnn, use_bi_direction=False)


def n_step_bigru(
        n_layers, dropout_ratio, hx, ws, bs, xs, train=True,
        use_cudnn=True):
    """Stacked Bi-directional Gated Recurrent Unit function.

    This function calculates stacked Bi-directional GRU with sequences.
    This function gets an initial hidden state :math:`h_0`, an input
    sequence :math:`x`, weight matrices :math:`W`, and bias vectors :math:`b`.
    This function calculates hidden states :math:`h_t` for each time :math:`t`
    from input :math:`x_t`.

    .. math::
       r^{f}_t &= \\sigma(W^{f}_0 x_t + W^{f}_3 h_{t-1} + b^{f}_0 + b^{f}_3)
       \\\\
       z^{f}_t &= \\sigma(W^{f}_1 x_t + W^{f}_4 h_{t-1} + b^{f}_1 + b^{f}_4)
       \\\\
       h^{f'}_t &= \\tanh(W^{f}_2 x_t + b^{f}_2 + r^{f}_t \\cdot (W^{f}_5
       h_{t-1} + b^{f}_5)) \\\\
       h^{f}_t &= (1 - z^{f}_t) \\cdot h^{f'}_t + z^{f}_t \\cdot h_{t-1}
       \\\\
       r^{b}_t &= \\sigma(W^{b}_0 x_t + W^{b}_3 h_{t-1} + b^{b}_0 + b^{b}_3)
       \\\\
       z^{b}_t &= \\sigma(W^{b}_1 x_t + W^{b}_4 h_{t-1} + b^{b}_1 + b^{b}_4)
       \\\\
       h^{b'}_t &= \\tanh(W^{b}_2 x_t + b^{b}_2 + r^{b}_t \\cdot (W^{b}_5
       h_{t-1} + b^{b}_5)) \\\\
       h^{b}_t &= (1 - z^{b}_t) \\cdot h^{b'}_t + z^{b}_t \\cdot h_{t-1}
       \\\\
       h_t  &= [h^{f}_t; h^{f}_t] \\\\

    where :math:`W^{f}` is weight matrices for forward-GRU, :math:`W^{b}` is
    weight matrices for backward-GRU.

    As the function accepts a sequence, it calculates :math:`h_t` for all
    :math:`t` with one call. Six weight matrices and six bias vectors are
    required for each layers. So, when :math:`S` layers exists, you need to
    prepare :math:`6S` weigth matrices and :math:`6S` bias vectors.

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
        ws (list of list of chainer.Variable): Weight matrices. ``ws[i]``
            represents weights for i-th layer.
            Each ``ws[i]`` is a list containing six matrices.
            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
            Only ``ws[0][j]`` where ``0 <= j < 3`` is ``(I, N)`` shape as they
            are multiplied with input variables. All other matrices has
            ``(N, N)`` shape.
        bs (list of list of chainer.Variable): Bias vectors. ``bs[i]``
            represnents biases for i-th layer.
            Each ``bs[i]`` is a list containing six vectors.
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
        use_bi_direction (bool): If ``True``, this function uses
            Bi-direction GRU.

    Returns:
        tuple: This functions returns a tuple concaining three elements,
            ``hy`` and ``ys``.
            - ``hy`` is an updated hidden states whose shape is same as ``hx``.
            - ``ys`` is a list of :class:`~chainer.Variable` . Each element
              ``ys[t]`` holds hidden states of the last layer corresponding
              to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is
              mini-batch size for time ``t``, and ``N`` is size of hidden
              units. Note that ``B_t`` is the same value as ``xs[t]``.

    """

    return n_step_gru_base(n_layers, dropout_ratio, hx, ws, bs, xs, train,
                           use_cudnn, use_bi_direction=True)


def n_step_gru_base(n_layers, dropout_ratio, hx, ws, bs, xs, train, use_cudnn,
                    use_bi_direction):
    """Base function for Stack GRU/BiGRU functions.

    This function is used at  :func:`chainer.functions.n_step_bigru` and
    :func:`chainer.functions.n_step_gru`.
    This function's behavior depends on argument ``use_bi_direction``.

    Args:
        n_layers(int): Number of layers.
        dropout_ratio(float): Dropout ratio.
        hx (chainer.Variable): Variable holding stacked hidden states.
            Its shape is ``(S, B, N)`` where ``S`` is number of layers and is
            equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
            dimention of hidden units.
        ws (list of list of chainer.Variable): Weight matrices. ``ws[i]``
            represents weights for i-th layer.
            Each ``ws[i]`` is a list containing six matrices.
            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
            Only ``ws[0][j]`` where ``0 <= j < 3`` is ``(I, N)`` shape as they
            are multiplied with input variables. All other matrices has
            ``(N, N)`` shape.
        bs (list of list of chainer.Variable): Bias vectors. ``bs[i]``
            represnents biases for i-th layer.
            Each ``bs[i]`` is a list containing six vectors.
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
        activation (str): Activation function name.
            Please select ``tanh`` or ``relu``.
        use_bi_direction (bool): If ``True``, this function uses
            Bi-direction GRU.

    .. seealso::
       :func:`chainer.functions.n_step_rnn`
       :func:`chainer.functions.n_step_birnn`

    """
    xp = cuda.get_array_module(hx, hx.data)

    if use_cudnn and xp is not numpy and cuda.cudnn_enabled and \
       _cudnn_version >= 5000:
        states = get_random_state().create_dropout_states(dropout_ratio)
        # flatten all input variables
        inputs = tuple(itertools.chain(
            (hx, ),
            itertools.chain.from_iterable(ws),
            itertools.chain.from_iterable(bs),
            xs))
        if use_bi_direction:
            rnn = NStepBiGRU(n_layers, states, train=train)
        else:
            rnn = NStepGRU(n_layers, states, train=train)

        ret = rnn(*inputs)
        hy, = ret[:1]
        ys = ret[1:]
        return hy, ys

    else:
        direction = 2 if use_bi_direction else 1
        hx = split_axis.split_axis(hx, n_layers * direction, axis=0,
                                   force_tuple=True)
        hx = [reshape.reshape(h, h.shape[1:]) for h in hx]

        xws = [concat.concat([w[0], w[1], w[2]], axis=0) for w in ws]
        hws = [concat.concat([w[3], w[4], w[5]], axis=0) for w in ws]
        xbs = [concat.concat([b[0], b[1], b[2]], axis=0) for b in bs]
        hbs = [concat.concat([b[3], b[4], b[5]], axis=0) for b in bs]

        xs_next = xs
        hy = []
        for layer in six.moves.range(n_layers):

            def _one_directional_loop(di):
                # di=0, forward GRU
                # di=1, backward GRU
                xs_list = xs_next if di == 0 else reversed(xs_next)
                layer_idx = direction * layer + di
                h = hx[layer_idx]
                h_list = []
                for x in xs_list:
                    batch = x.shape[0]
                    if h.shape[0] > batch:
                        h, h_rest = split_axis.split_axis(h, [batch], axis=0)
                    else:
                        h_rest = None

                    if layer > 0:
                        x = dropout.dropout(x, ratio=dropout_ratio,
                                            train=train)

                    gru_x = linear.linear(x, xws[layer_idx], xbs[layer_idx])
                    gru_h = linear.linear(h, hws[layer_idx], hbs[layer_idx])

                    W_r_x, W_z_x, W_x = split_axis.split_axis(gru_x, 3, axis=1)
                    U_r_h, U_z_h, U_x = split_axis.split_axis(gru_h, 3, axis=1)

                    r = sigmoid.sigmoid(W_r_x + U_r_h)
                    z = sigmoid.sigmoid(W_z_x + U_z_h)
                    h_bar = tanh.tanh(W_x + r * U_x)
                    h_bar = (1 - z) * h_bar + z * h
                    if h_rest is not None:
                        h = concat.concat([h_bar, h_rest], axis=0)
                    else:
                        h = h_bar
                    h_list.append(h_bar)
                return h, h_list

            # Forward GRU
            h, h_forward = _one_directional_loop(di=0)
            hy.append(h)

            if use_bi_direction:
                # Backward GRU
                h, h_backward = _one_directional_loop(di=1)
                h_backward.reverse()
                # Concat
                xs_next = [concat.concat([hfi, hbi], axis=1) for (hfi, hbi) in
                           six.moves.zip(h_forward, h_backward)]
                hy.append(h)
            else:
                # Uni-directional GRU
                xs_next = h_forward

        ys = xs_next
        hy = stack.stack(hy)
        return hy, tuple(ys)
