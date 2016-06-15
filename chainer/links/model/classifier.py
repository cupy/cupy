from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link


class Classifier(link.Chain):

    """A simple classifier model.

    This is an example of chain that wraps another chain. It computes the
    loss and accuracy based on a given input/label pair.

    Args:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.

    Attributes:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        y (~chainer.Variable): Prediction for the last minibatch.
        loss (~chainer.Variable): Loss value for the last minibatch.
        accuracy (~chainer.Variable): Accuracy for the last minibatch.
        compute_accuracy (bool): If ``True``, compute accuracy on the forward
            computation. The default value is ``True``.

    """

    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy):
        super(Classifier, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.y = None
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        """Computes the loss value for an input and label pair.

        It also computes accuracy and stores it to the attribute.

        Args:
            x (~chainer.Variable): Input minibatch.
            t (~chainer.Variable): Corresponding ground truth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(x)
        self.loss = self.lossfun(self.y, t)
        if self.compute_accuracy:
            self.accuracy = accuracy.accuracy(self.y, t)
        return self.loss
