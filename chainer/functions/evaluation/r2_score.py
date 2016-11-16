from chainer import cuda
from chainer import function
from chainer.utils import type_check


class R2_score(function.Function):
    def __init__(self, sample_weight, multioutput):
        if sample_weight is not None:
            raise NotImplementedError()
        if multioutput in ['uniform_average', 'raw_values']:
            self.multioutput = multioutput
        else:
            raise ValueError("invalid multioutput argument")

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        pred_type, true_type = in_types

        type_check.expect(
            pred_type.dtype.kind == 'f',
            true_type.dtype.kind == 'f'
        )

        type_check.expect(
            pred_type.ndim >= true_type.ndim,
            pred_type.shape == true_type.shape,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        pred, true = inputs
        SS_res = xp.sum((pred-true)**2, axis=0)
        SS_tot = xp.sum((true-xp.mean(true, axis=0))**2, axis=0)
        if self.multioutput == 'uniform_average':
            return xp.asarray((1 - SS_res / SS_tot).mean(), dtype=pred.dtype),
        elif self.multioutput == 'raw_values':
            return xp.asarray((1 - SS_res / SS_tot), dtype=pred.dtype),


def r2_score(pred, true, sample_weight=None, multioutput='uniform_average'):
    return R2_score(sample_weight=sample_weight, multioutput=multioutput)\
            (pred, true)
