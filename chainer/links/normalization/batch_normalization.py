import numpy

from chainer import cuda
from chainer.functions.normalization import batch_normalization
from chainer import initializers
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
    user must use this mode to feed mini-batches running through whole training
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
        use_gamma (bool): If ``True``, use scaling parameter. Otherwise, use
            unit(1) which makes no effect.
        use_beta (bool): If ``True``, use shifting parameter. Otherwise, use
            unit(0) which makes no effect.
        use_cudnn (bool): If ``True``, then this link uses cuDNN if available.

    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_

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
        use_cudnn (bool): If ``True``, then this link uses cuDNN if available.

    """

    def __init__(self, size, decay=0.9, eps=2e-5, dtype=numpy.float32,
                 use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None, use_cudnn=True):
        super(BatchNormalization, self).__init__()
        if use_gamma:
            self.add_param('gamma', size, dtype=dtype)
            if initial_gamma is None:
                initial_gamma = initializers.One()
            initializers.init_weight(self.gamma.data, initial_gamma)
        if use_beta:
            self.add_param('beta', size, dtype=dtype)
            if initial_beta is None:
                initial_beta = initializers.Zero()
            initializers.init_weight(self.beta.data, initial_beta)
        self.add_persistent('avg_mean', numpy.zeros(size, dtype=dtype))
        self.add_persistent('avg_var', numpy.zeros(size, dtype=dtype))
        self.add_persistent('N', 0)
        self.decay = decay
        self.eps = eps
        self.use_cudnn = use_cudnn

    def __call__(self, x, test=False, finetune=False):
        """Invokes the forward propagation of BatchNormalization.

        BatchNormalization accepts additional arguments, which controls three
        different running mode.

        Args:
            x (Variable): Input variable.
            test (bool): If ``True``, BatchNormalization runs in testing mode;
                it normalizes the input using pre-computed statistics.
            finetune (bool): If ``finetune`` is ``True`` and ``test`` is
                ``False``, BatchNormalization runs in fine-tuning mode; it
                accumulates the input array to compute population statistics
                for normalization, and normalizes the input using batch
                statistics.

        If ``test`` is ``False``, then BatchNormalization runs in training
        mode; it computes moving averages of mean and variance for evaluation
        during training, and normalizes the input using batch statistics.

        """
        if hasattr(self, 'gamma'):
            gamma = self.gamma
        else:
            with cuda.get_device(self._device_id):
                gamma = variable.Variable(self.xp.ones(
                    self.avg_mean.shape, dtype=x.dtype), volatile='auto')
        if hasattr(self, 'beta'):
            beta = self.beta
        else:
            with cuda.get_device(self._device_id):
                beta = variable.Variable(self.xp.zeros(
                    self.avg_mean.shape, dtype=x.dtype), volatile='auto')

        if not test:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            func = batch_normalization.BatchNormalizationFunction(
                self.eps, self.avg_mean, self.avg_var, True, decay,
                self.use_cudnn)
            ret = func(x, gamma, beta)

            self.avg_mean[:] = func.running_mean
            self.avg_var[:] = func.running_var
        else:
            # Use running average statistics or fine-tuned statistics.
            mean = variable.Variable(self.avg_mean, volatile='auto')
            var = variable.Variable(self.avg_var, volatile='auto')
            ret = batch_normalization.fixed_batch_normalization(
                x, gamma, beta, mean, var, self.eps, self.use_cudnn)
        return ret

    def start_finetuning(self):
        """Resets the population count for collecting population statistics.

        This method can be skipped if it is the first time to use the
        fine-tuning mode. Otherwise, this method should be called before
        starting the fine-tuning mode again.

        """
        self.N = 0
