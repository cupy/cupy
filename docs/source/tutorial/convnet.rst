Convolutional Network for Visual Recognition Tasks
--------------------------------------------------

.. currentmodule:: chainer

In this section, you will learn how to write

* Convolutional network by defining a :class:`~chainer.Chain` object,
* Convolutional network that has a large number of components easily

After reading this section, you will be able to:

* Write your original convolutional network in Chainer

Convolutional Network
~~~~~~~~~~~~~~~~~~~~~

A convolutional network (ConvNet) is comprised of convolution layers.
This type of network is oftenly used for visual recognition tasks, e.g., classifying hand-written digits or natural images into known classes, detectiong objects from an image, and labeling all pixels of a image into known classes (semantic segmenation), and so on.
A typical ConvNet takes a set of images whose shape is :math:`(N, C, H, W)`, where :math:`N` denotes the number of images in a mini-batch, :math:`C` denotes the number of channels of those images, :math:`H` and :math:`W` denote the height and width of those images, respectively.
Then, it outputs a fixed-sized vector for image classification tasks, or an image that has the corresponding size to the input for a pixel labeling task, etc.

Here, let's start from defining [^LeNet5] in Chainer. This is to classify hand-written digit images. It comprises 7 layers (conv - mpool - conv - mpool - conv - fc - fc).
In Chainer, the model can be written as follows:

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
            h = self.conv1(x)
            h = F.sigmoid(h)
            h = F.max_pooling_2d(h, 2, 2)
            h = self.conv2(x)
            h = F.sigmoid(h)
            h = F.max_pooling_2d(h, 2, 2)
            h = self.conv3(x)
            h = F.sigmoid(h)
            h = self.fc4(x)
            h = F.sigmoid(h)
            h = self.fc5(x)
            loss = F.softmax_cross_entropy(h, t)


[^LeNet5]: Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278–2324, 1998.
