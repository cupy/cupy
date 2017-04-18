Convolutional Network for Visual Recognition Tasks
``````````````````````````````````````````````````

.. currentmodule:: chainer

In this section, you will learn how to write

* A small convolutional network with a model class that is inherited from :class:`~chainer.Chain`,
* A large convolutional network that has several building block networks with :class:`~chainer.ChainList`.

After reading this section, you will be able to:

* Write your original convolutional network in Chainer

Convolutional Network
'''''''''''''''''''''

A convolutional network (ConvNet) is mainly comprised of convolutional layers.
This type of network is oftenly used for various visual recognition tasks,
e.g., classifying hand-written digits or natural images into given object
classes, detectiong objects from an image, and labeling all pixels of an image
with the object classes (semantic segmenation), and so on.

In such tasks, a typical ConvNet takes a set of images whose shape is
:math:`(N, C, H, W)`, where

- :math:`N` denotes the number of images in a mini-batch,
- :math:`C` denotes the number of channels of those images,
- :math:`H` and :math:`W` denote the height and width of those images,

respectively. Then, it typically outputs a fixed-sized vector as membership
probability over the target object classes. It also can output a set of feature
maps that have the corresponding size to the input image for a pixel labeling
task, etc.

LeNet5
......

Here, let's start from defining LeNet5 [LeCun98]_ in Chainer.
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

You can take another way to do the same thing. For example,
:meth:`~chainer.Chain.add_link` of :class:`~chainer.Chain` class enables to
register the trainable layers (i.e., :class:`~chainer.Link` s) to the model, so
that the above :meth:`__init__` can also be written as follows:

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

The model class is instantiated before forward and backward computations.
To give input images and label vectors simply by calling the model object
like a function, :meth:`__call__` is usually defined in the model class.
This method performs the forward computation of the model. Chainer has the
strong autograd system for any computational graphs written with
:class:`~chainer.Function`s and :class:`~chainer.Links`s (actually a
:class:`~chainer.Links` calls a corresponding :class:`~chainer.Function` inside
of it), so that you don't need to write any codes for backward computations in
the model. Just prepare the data, then give it to the model. Then the resulting
:class:`~chainer.Variable` from the forward computation has
:meth:`~chainer.Variable.backward` method to perform autograd. In the above
model, :meth:`__call__` has a ``if`` statement at the end to switch its
behavior by the model's running mode, i.e., training mode or not. When it's in
training mode, this method returns a loss value, otherwise it returns a
prediction result.

If you don't want to write ``conv1`` and the other layers more than twice, you
can also write the model like in this way:

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

This code creates a list that contains all :class:`~chainer.Link` s and
:class:`~chainer.Function` s first right after calling its superclass's
constructor. Then the elements of the list are registered to this model as
trainable layers when the name of an element doesn't start with ``_``
character. This operation can be freely replaced with many other ways because
those names are just designed to select :class:`~chainer.Link` s only from the
list ``net`` easily. :class:`~chainer.Function` doesn't have any trainable
parameters, so that we can't register it to the model with
:meth:`~chainer.Chain.add_link`, but we want to use
:class:`~chainer.Function` s also for constructing a forward path. The list
``net`` is stored to the attribute attr:`forward` to refer it in
:meth:`__call__`. In :meth:`__call__`, it retrieves all layers in the network
from :attr:`forward` sequentially regardless of what types of object (
:class:`~chainer.Link` or :class:`~chainer.Function`) it is, and gives the
input variable or the intermediate output from the previous layer to the
current layer. The last part of the :meth:`__call__` to switch its behavior
by the training/inference mode is the same as the former way.

Chainer is flexible, so we can write our original network in many different
ways. It would give users an intuitive way to design brand-new and complex
models.

.. [LeCun98] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner.
    Gradient-based learning applied to document recognition. Proceedings of the
    IEEE, 86(11), 2278–2324, 1998.

VGG16 and ResNet
................

Next, let's write more large models like VGG16 [Simonyan14]_ and ResNet [He16]_
in Chainer. To write a large network consisted of several building block
networks, :class:`~chainer.ChainList` is useful.


.. [Simonyan14] Simonyan, K. and Zisserman, A., Very Deep Convolutional
    Networks for Large-Scale Image Recognition. arXiv preprint arXiv:1409.1556,
    2014.

.. [He16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual
    Learning for Image Recognition. The IEEE Conference on Computer Vision and
    Pattern Recognition (CVPR), pp. 770-778, 2016.
