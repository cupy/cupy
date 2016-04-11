import numpy

from chainer import cuda
from chainer.functions.normalization import batch_normalization
from chainer import link
from chainer import variable


class BatchNormalization(link.Link):

    """Batch normalization layer on outputs of linear or convolution functions.

    This link wraps the :func:`~chainer.functions.batch_normalization` and
    :func:`~chainer.functions.fixed_batch_normalization` functions.

    It runs in three modes: training mode, fine-tuning mode, and testing mode.

    In training mode, it normalizes the input by *batch statistics*. It also
    maintains approximated population statistics by moving averages, which can
    be used for instant evaluation in testing mode.

    In fine-tuning mode, it accumulates the input to compute *population
    statistics*. In order to correctly compute the population statistics, a
    user must use this mode to feed mini batches running through whole training
    dataset.

    In testing mode, it uses pre-computed population statistics to normalize
    the input variable. The population statistics is approximated if it is
    computed by training mode, or accurate if it is correctly computed by
    fine-tuning mode.

    Args:
        size (int or tuple of ints): Size (or shape) of channel
            dimensions.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.

    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <http://arxiv.org/abs/1502.03167>`_

    .. seealso::
       :func:`~chainer.functions.batch_normalization`,
       :func:`~chainer.functions.fixed_batch_normalization`

    Attributes:
        gamma (~chainer.Variable): Scaling parameter.
        beta (~chainer.Variable): Shifting parameter.
        avg_mean (~chainer.Variable): Population mean.
        avg_var (~chainer.Variable): Population variance.
        N (int): Count of batches given for fine-tuning.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability. This value is added
            to the batch variances.

    """
    def __init__(self, size, decay=0.9, eps=1e-5, dtype=numpy.float32):
        super(BatchNormalization, self).__init__()
        self.add_param('gamma', size, dtype=dtype)
        self.gamma.data.fill(1)
        self.add_param('beta', size, dtype=dtype)
        self.beta.data.fill(0)
        self.add_persistent('avg_mean', numpy.zeros(size, dtype=dtype))
        self.add_persistent('avg_var', numpy.zeros(size, dtype=dtype))
        self.add_persistent('N', 0)
        self.decay = decay
        self.eps = eps

    def __call__(self, x, test=False, finetune=False):
        """Invokes the forward propagation of BatchNormalization.

        BatchNormalization accepts additional arguments, which controls three
        different running mode.

        Args:
            x (Variable): An input variable.
            test (bool): If ``True``, BatchNormalization runs in testing mode;
                it normalizes the input using pre-computed statistics.
            finetune (bool): If ``True``, BatchNormalization runs in
                fine-tuning mode; it accumulates the input array to compute
                population statistics for normalization, and normalizes the
                input using batch statistics.

        If ``test`` and ``finetune`` are both ``False``, then
        BatchNormalization runs in training mode; it computes moving averages
        of mean and variance for evaluation during training, and normalizes the
        input using batch statistics.

        """
        use_batch_mean = not test or finetune

        if use_batch_mean:
            func = batch_normalization.BatchNormalizationFunction(self.eps)
            ret = func(x, self.gamma, self.beta)

            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            with cuda.get_device(x.data):
                m = x.data.size // self.gamma.data.size
                adjust = m / max(m - 1., 1.)  # unbiased estimation
                self.avg_mean *= decay
                func.mean *= 1 - decay  # reuse buffer as a temporary
                self.avg_mean += func.mean
                del func.mean
                self.avg_var *= decay
                func.var *= (1 - decay) * adjust  # reuse buffer as a temporary
                self.avg_var += func.var
                del func.var
        else:
            mean = variable.Variable(self.avg_mean, volatile='auto')
            var = variable.Variable(self.avg_var, volatile='auto')
            ret = batch_normalization.fixed_batch_normalization(
                x, self.gamma, self.beta, mean, var, self.eps)
        return ret

    def start_finetuning(self):
        """Resets the population count for collecting population statistics.

        This method can be skipped if it is the first time to use the
        fine-tuning mode. Otherwise, this method should be called before
        starting the fine-tuning mode again.

        """
        self.N = 0
