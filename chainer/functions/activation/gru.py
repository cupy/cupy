from chainer import function_set
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.connection import linear


class GRU(function_set.FunctionSet):

    def __init__(self, n_unit):
        super(GRU, self).__init__(
            W_r=linear.Linear(n_unit, n_unit),
            U_r=linear.Linear(n_unit, n_unit),
            W_z=linear.Linear(n_unit, n_unit),
            U_z=linear.Linear(n_unit, n_unit),
            W=linear.Linear(n_unit, n_unit),
            U=linear.Linear(n_unit, n_unit),
        )

    def __call__(self, h, x):
        r = sigmoid.sigmoid(self.W_r(x) + self.U_r(h))
        z = sigmoid.sigmoid(self.W_z(x) + self.U_z(h))
        h_bar = tanh.tanh(self.W(x) + self.U(r * h))
        h_new = (1 - z) * h + z * h_bar
        return h_new
