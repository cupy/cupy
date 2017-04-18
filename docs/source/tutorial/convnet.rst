Convolutional Network for Visual Recognition Tasks
``````````````````````````````````````````````````

.. currentmodule:: chainer

In this section, you will learn how to write

* A small convolutional network with a model class that is inherited from :class:`~chainer.Chain`,
* A large convolutional network that has several building block networks with :class:`~chainer.ChainList`.

After reading this section, you will be able to:

* Write your original convolutional network in Chainer

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
''''''

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

        def __call__(self, x):
            h = F.sigmoid(self.conv1(x))
            h = F.max_pooling_2d(h, 2, 2)
            h = F.sigmoid(self.conv2(h))
            h = F.max_pooling_2d(h, 2, 2)
            h = F.sigmoid(self.conv3(h))
            h = F.sigmoid(self.fc4(h))
            if self.train:
                return self.fc5(h)
            return F.softmax(self.fc5(h))

A typical way to write your network is creating a new class inherited from
:class:`~chainer.Chain` class. When defining your model in this way, typically,
all the layers which have trainable parameters are registered to the model
by giving the objects of :class:`~chainer.Link` to the superclass's constructer
as keyword arguments (see the above :meth:`__init__`).

You can take another way to do the same thing. For example,
:meth:`~chainer.Chain.add_link` of :class:`~chainer.Chain` class enables to
register the trainable layers (i.e., :class:`~chainer.Link` s) to the model, so
that the above :meth:`__init__` can also be written as follows:

.. code-block:: python

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
training mode, this method returns the output value of the last layer as is to
compute the loss later on, otherwise it returns a prediction result by
calculating :meth:`~chainer.functions.softmax`.

If you don't want to write ``conv1`` and the other layers more than once, you
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

        def __call__(self, x):
            for f in self.forward:
                x = getattr(self, f[0])(x)
            if self.train:
                return x
            return F.softmax(x)

This code creates a list of all :class:`~chainer.Link` s and
:class:`~chainer.Function` s after calling its superclass's constructor.
Then the elements of the list are registered to this model as
trainable layers when the name of an element doesn't start with ``_``
character. This operation can be freely replaced with many other ways because
those names are just designed to select :class:`~chainer.Link` s only from the
list ``net`` easily. :class:`~chainer.Function` doesn't have any trainable
parameters, so that we can't register it to the model with
:meth:`~chainer.Chain.add_link`, but we want to use
:class:`~chainer.Function` s for constructing a forward path. The list
``net`` is stored as an attribute attr:`forward` to refer it in
:meth:`__call__`. In :meth:`__call__`, it retrieves all layers in the network
from :attr:`self.forward` sequentially regardless of what types of object (
:class:`~chainer.Link` or :class:`~chainer.Function`) it is, and gives the
input variable or the intermediate output from the previous layer to the
current layer. The last part of the :meth:`__call__` to switch its behavior
by the training/inference mode is the same as the former way.

Ways to calculate loss
......................

When you train the model with label vector ``t``, the loss should be calculated
using the output from the model. There also are several ways to calculate the
loss:

.. doctest::

    model = LeNet5()

    # Input data and label
    x = np.random.rand(32, 1, 28, 28).astype(np.float32)
    t = np.random.randint(0, 10, size=(32,)).astype(np.int32)

    # Forward computation
    y = model(x)

    # Loss calculation
    loss = F.softmax_cross_entropy(y, t)

This is a primitive way to calculate a loss value from the output of the model.
On the other hand, the loss computation can be included in the model itself by
wrapping the model object (:class:`~chainer.Chain` or
:class:`~chainer.ChainList` object) with a class inherited from
:class:`~chainer.Chain`. The outer :class:`~chainer.Chain` should take the
model defined above and register it through the constructor of its superclass
or :meth:`~chainer.Chain.add_link`. :class:`~chainer.Chain` is actually
inherited from :class:`~chainer.Link`, so that :class:`~chainer.Chain` itself
can also be registedred as a trainable :class:`~chainer.Link` to another
:class:`~chainer.Chain`. Actually, :class:`~chainer.links.Classifier` class to
wrap the model and add the loss computation to the model already exists.
Using this, the loss computation can be implanted to the model itself by this
way:

.. doctest::

    model = L.Classifier(LeNet5())

    # Foward & Loss calculation
    loss = model(x, t)

This class takes a model object as an iput argument and registers it to
``predictor`` property as a trained parameter. Then calling the returned object
as a function like the above code with feeding ``x`` and ``t`` as the input
arguments, yield a loss value calculated from those two input variables.

See the detailed implementation of :class:`~chainer.links.Classifier` from
here: :class:`chainer.links.Classifier` and check the implementation by looking
at the source.

Chainer is flexible, so we can write our original network in many different
ways. It would give users an intuitive way to design brand-new and complex
models.

VGG16
'''''

Next, let's write more large models like VGG16 [Simonyan14]_ in Chainer.
When you write a large network consisted of several building block
networks, :class:`~chainer.ChainList` is useful. First, let's see how to write
a VGG16 model:

.. doctest::

    class VGG16(chainer.ChainList):

        def __init__(self):
            w = chainer.initializers.HeNormal()
            super(VGG16, self).__init__(
                VGGBlock(64),
                VGGBlock(128),
                VGGBlock(256, 3),
                VGGBlock(512, 3),
                VGGBlock(512, 3, True))
            self.train = True

        def __call__(self, x):
            for f in self.children():
                x = f(x, self.train)
            if self.train:
                return x
            return F.softmax(x)

    class VGGBlock(chainer.Chain):

        def __init__(self, n_channels, n_convs=2, fc=False):
            w = chainer.initializers.HeNormal()
            super(VGG16Block, self).__init__(
                conv1=L.Convolution2D(None, n_channels, 3, 1, 1, initialW=w),
                conv2=L.Convolution2D(
                    n_channels, n_channels, 3, 1, 1, initialW=w))
            if n_convs == 3:
                self.add_link('conv3', L.Convolution2D(
                    n_channels, n_channels, 3, 1, 1, initialW=w))
            if fc:
                self.add_link('fc4', L.Linear(None, 4096, initialW=w))
                self.add_link('fc5', L.Linear(4096, 4096, initialW=w))
                self.add_link('fc6', L.Linear(4096, 1000, initialW=w))

            self.n_convs = n_convs
            self.fc = fc

        def __call__(self, x, train):
            h = F.relu(self.conv1(x))
            h = F.relu(self.conv2(h))
            if self.n_convs == 3:
                h = F.relu(self.conv3(h))
            h = F.max_pooling_2d(h, 2, 2)
            if self.fc:
                h = F.dropout(F.relu(self.fc4(h)), train=train)
                h = F.dropout(F.relu(self.fc5(h)), train=train)
                h = self.fc6(h)
            return h

That's it. VGG16 is a model which won the 1st place in [classification +
localization task at ILSVRC 2014](http://www.image-net.org/challenges/LSVRC/2014/results#clsloc),
and since then, became one of standard models for many different tasks as a
pre-trained model. This has 16-layers, so it's called "VGG-16", but we can
write this model without writing all layers independently. Because this model
is consisted of several building blocks that have the same architecture,
so that we can build the whole network by re-using the building block
definition. Each part of the network is consisted of 2 or 3 convolutional
layers and activation function (:meth:`~chainer.functions.relu`) following
them, and :meth:`~chainer.functions.max_pooling_2d` operations. This block
is written as :class:`VGGBlock` in the above example code. And the whole
network just calls this block one by one in sequential manner.

ResNet152
'''''''''

How about ResNet? ResNet [He16]_ came in the next year's ILSVRC. It is way deeper
model than VGG16, so the depth is up to 152 layers. This sounds super laboring
to build, but it can be implemented in almost same manner as VGG16. In the
other words, it's easy. One possible way to write ResNet-152 is:

.. doctest::

    class ResNet152(chainer.ChainList):

        def __init__(self, n_blocks=[3, 8, 36, 3]):
            super(ResNet152, self).__init__(
                conv1=L.Convolution2D(
                    None, 64, 7, 2, 3, initialW=w, nobias=True),
                bn1=L.BatchNormalization(64),
                res2=ResBlock(n_blocks[0], 64, 64, 256, 1),
                res3=ResBlock(n_blocks[1], 256, 128, 512),
                res4=ResBlock(n_blocks[2], 512, 256, 1024),
                res5=ResBlock(n_blocks[3], 1024, 512, 2048))
            self.train = True

        def __call__(self, x):
            h = self.bn1(self.conv1(x), test=not self.train)
            h = F.max_pooling_2d(F.relu(h), 2, 2)
            h = self.res2(h)
            h = self.res3(h)
            h = self.res4(h)
            h = self.res5(h)
            if self.train:
                return h
            return F.softmax(h)

    class ResBlock(chainer.ChainList):

        def __init__(self, n_layers, n_in, n_mid, n_out, stride=2):
            w = chainer.initializers.HeNormal()
            super(ResBlock, self).__init__()
            self.add_link(BottleNeck(n_in, n_mid, n_out, stride, True))
            for _ in range(n_layers - 1):
                self.add_link(BottleNeck(n_out, n_mid, n_out))

        def __call__(self, x, train):
            for f in self.children():
                x = f(x, train)
            return x

    class BottleNeck(chainer.Chain):

        def __init__(self, n_in, n_mid, n_out, stride=1, proj=False):
            w = chainer.initializers.HeNormal()
            super(BottleNeck, self).__init__(
                conv1x1a=L.Convolution2D(
                    n_in, n_mid, 1, stride, 0, initialW=w, nobias=True),
                conv3x3b=L.Convolution2D(
                    n_mid, n_mid, 3, 1, 1, initialW=w, nobias=True),
                conv1x1c=L.Convolution2D(
                    n_mid, n_out, 1, 1, 0, initialW=w, nobias=True),
                bn_a=L.BatchNormalization(n_mid),
                bn_b=L.BatchNormalization(n_mid),
                bn_c=L.BatchNormalization(n_out))
            if proj:
                self.add_link('conv1x1r', L.Convolution2D(
                    n_in, n_out, 1, stride, 0, initialW=w, nobias=True))
                self.add_link('bn_r', L.BatchNormalization(n_out))
            self.proj = proj

        def __call__(self, x, train):
            h = F.relu(self.bn_a(self.conv1x1a(x), test=not train))
            h = F.relu(self.bn_b(self.conv3x3b(x), test=not train))
            h = self.bn_c(self.conv1x1c(x), test=not train)
            if self.proj:
                x = self.bn_r(self.conv1x1r(x), test=not train)
            return F.relu(h + x)

In :class:`BottleNeck` class, it switches whther ``conv1x1r`` to fit the
channel dimension of the input ``x`` and the output ``h`` and ``bn_r`` for it
should be added or not with respect to :attr:`~BottleNeck.proj` attribute.
Writing the building block in this way gains the re-usability of a class a lot.
It switches not only the behavior in :meth:`__class__` by flags but also the
parameter registration. In this case, when :attr:`proj` is ``False``, the
:class:`BottleNeck` doesn't have `conv1x1r` and `bn_r` layers, so the memory
usage would be efficient compared to the case when it registers both anyway and
just ignore them if :attr:`proj` is ``False``.

Using nesting :class:`~chainer.Chain` s and :class:`~chainer.ChainList` for
sequential part enables us to write complex and very deep models easily.

Use Pre-trained Models
''''''''''''''''''''''

Various ways to write your models is described above. And VGG16 and ResNet are
very useful as general feature extractor for many different tasks from image
classification. So Chainer provides you the pre-trained models of VGG16 and
ResNet-50/101/152 models with a simple API. You can use those models in this
way:

.. doctest::

    from chainer.links import VGG16Layers

    model = VGG16Layers()

When :class:`~chainer.links.VGG16Layers` is instantiated, the pre-trained
parameters is automatically downloaded from the author's server. So you can
immediately start to use VGG16 with pre-trained weight as a good image feature
extractor. See the details of this model here:
:class:`chainer.links.VGG16Layers`.

In the case of ResNet models, there are 3 variation in terms of the number of
layers. We have :class:`chainer.links.ResNet50`,
:class:`chainer.links.ResNet101`, and :class:`chainer.links.ResNet152` models
with easy parameter loading feature. ResNet's pre-trained parameters are not
available for direct downloading, so you need to download the weight from the
author's web page first, and then place it into the dir
``$CHAINER_DATSET_ROOT/pfnet/chainer/models`` or your favorite place. Once
the preparation is finished, the usage is the same as VGG16:

.. doctest::

    from chainer.links import ResNet152Layers

    model = ResNet152layers()

Please see the details of usage and how to prepare the pre-trained weights for
ResNet here: :class:`chainer.links.ResNet50`

References
..........

.. [LeCun98] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner.
    Gradient-based learning applied to document recognition. Proceedings of the
    IEEE, 86(11), 2278–2324, 1998.
.. [Simonyan14] Simonyan, K. and Zisserman, A., Very Deep Convolutional
    Networks for Large-Scale Image Recognition. arXiv preprint arXiv:1409.1556,
    2014.
.. [He16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual
    Learning for Image Recognition. The IEEE Conference on Computer Vision and
    Pattern Recognition (CVPR), pp. 770-778, 2016.
