Convolutional Network for Visual Recognition Tasks
--------------------------------------------------

.. currentmodule:: chainer

In this section, you will learn how to write

* A small convolutional network with a model class that is inherited from :class:`~chainer.Chain`,
* A large convolutional network that has several building block networks with :class: `~chainer.ChainList`.

After reading this section, you will be able to:

* Write your original convolutional network in Chainer

Convolutional Network
~~~~~~~~~~~~~~~~~~~~~

A convolutional network (ConvNet) is comprised of convolution layers.
This type of network is oftenly used for various visual recognition tasks,
e.g., classifying hand-written digits or natural images into given object
classes, detectiong objects from an image, and labeling all pixels of a image
with the object classes (semantic segmenation), and so on.

In such tasks, a typical ConvNet takes a set of images whose shape is
:math:`(N, C, H, W)`, where

- :math:`N` denotes the number of images in a mini-batch,
- :math:`C` denotes the number of channels of those images,
- :math:`H` and :math:`W` denote the height and width of those images,

respectively. Then, it outputs a fixed-sized vector as membership probability
over target object classes. It also can output a set of feature maps that have
the corresponding size to the input image for a pixel labeling task, etc.

Here, let's start from defining [LeCun98]_ in Chainer.
This is a ConvNet model that has 5 layers comprised of 3 convolutional layers
and 2 fully-connected layers. This has been proposed to classify hand-written
digit images in 1998. In Chainer, the model can be written as follows:

.. testcode::

    class LeNet5(Chain):
        def __init__(self):
            super(ConvNet, self).__init__(
                conv1=L.Convolution2D(
                    in_channels=1, out_channels=6, ksize=5, stride=1),
                conv2=L.Convolution2D(
                    in_channels=6, out_channels=16, ksize=5, stride=1),
                conv3=L.Convolution2D(
                    in_channels=16, out_channels=120, ksize=5, stride=1),
                fc4=L.Linear(None, 84),
                fc5=L.Linear(84, 10),
            )
            self.train = True

        def __call__(self, x, t):
            h = F.sigmoid(self.conv1(x))
            h = F.max_pooling_2d(h, 2, 2)
            h = F.sigmoid(self.conv2(x))
            h = F.max_pooling_2d(h, 2, 2)
            h = F.sigmoid(self.conv3(x))
            h = F.sigmoid(self.fc4(x))
            h = self.fc5(x)
            if self.train:
                loss = F.softmax_cross_entropy(h, t)
                return loss
            pred = F.softmax(h)
            return pred

A typical way to write your network is creating a new class inherited from
:class:`~chainer.Chain` class. When defining your model in this way, typically,
all the layers which have trainable parameters are registered to the model
by giving the objects of :class:`~chainer.Link` to the superclass's constructer
as keyword arguments (see the above :meth:`__init__`).

This can also be done in other ways. For example,
:meth:`~chainer.Chain.add_link` of :class:`~chainer.Chain` class enables to
register the trainable layers (:class:`~chainer.Link` s) to the model, so that
the above :meth:`__init__` can also be:

.. testcode::

    def __init__(self):
        self.add_link('conv1', L.Convolution2D(1, 6, 5, 1))
        self.add_link('conv2', L.Convolution2D(6, 16, 5, 1))
        self.add_link('conv3', L.Convolution2D(16, 120, 5, 1))
        self.add_link('fc4', L.Linear(None, 84))
        self.add_link('fc5', L.Linear(84, 10))
        self.train = True

(Argments to :class:`~chainer.links.Convolution2D` are given without keywords
here for simplicity.)

The model class is instantiated before training and also before inference.
To give input images and label vectors simply by calling the model object
like a function, :meth:`__call__` is defined in the model typically. This
method performs the forward computation of the model. Chainer has the strong
autograd system for any computational graphs written with
:class:`~chainer.Function`s and :class:`~chainer.Links`s (actually a
:class:`~chainer.Links` calls a corresponding :class:`~chainer.Function` inside
of it), so that you don't need to write any backward computation codes for the
model. Just prepare the data, then give it to the model. Then the
returned :class:`~chainer.Variable` has :meth:`~chainer.Variable.backward`
method to perform autograd. In the above model, :meth:`__call__` has a ``if``
statement at the end to switch its behavior by the model's running mode, i.e.,
training mode or not. When it's in training mode, this method returns the loss,
otherwise it returns the prediction.

If you don't want to write ``conv1`` and the other layers never, you can also
write the model like in this way:

.. testcode::

    class LeNet5(Chain):
        def __init__(self):
            super(ConvNet, self).__init__()
            net  = [('conv1',   L.Convolution2D(1, 6, 5, 1))]
            net += [('_mpool1', F.MaxPooling2D(2, 2))]
            net += [('_sigm1',  F.Sigmoid())]
            net += [('conv2',   L.Convolution2D(6, 16, 5, 1))]
            net += [('_mpool2', F.MaxPooling2D(2, 2))]
            net += [('_sigm2',  F.Sigmoid())]
            net += [('conv3',   L.Convolution2D(6, 16, 5, 1))]
            net += [('_mpool3', F.MaxPooling2D(2, 2))]
            net += [('_sigm3',  F.Sigmoid())]
            net += [('fc4',     L.Linear(None, 84))]
            net += [('_sigm4',  F.Sigmoid())]
            net += [('fc5',     L.Linear(84, 10))]
            net += [('_sigm5',  F.Sigmoid())]
            for n in net:
                if not n[0].startwith('_'):
                    self.add_link(n)
            self.forward = net
            self.train = True

        def __call__(self, x, t):
            for f in self.forward:
                x = getattr(self, f[0])(x)
            if self.train:
                loss = F.softmax_cross_entropy(x, t)
                return loss
            pred = F.softmax(x)
            return pred



.. [LeCun98] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner.
    Gradient-based learning applied to document recognition. Proceedings of the
    IEEE, 86(11), 2278–2324, 1998.
