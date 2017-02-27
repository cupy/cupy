Convolutional Network for Visual Recognition Tasks
--------------------------------------------------

.. currentmodule:: chainer

In this section, you will learn how to write

* Convolutional network by defining a :class:`~chainer.Chain` object,
* Convolutional network that has a large number of components easily

After reading this section, you will be able to:

* Write your original convolutional network

Convolutional Network
~~~~~~~~~~~~~~~~~~~~~

A convolutional network (ConvNet) is comprised of convolution layers.
This type of network is oftenly used for visual recognition tasks, e.g., classifying hand-written digits, natural images into known classes, detectiong objects from an image, and labeling all pixels of a image, and so on.
A typical ConvNet takes a set of images whose shape is :math:`(N, C, H, W)`, where :math:`N` denotes the number of images in a mini-batch, :math:`C` denotes the number of channels of those images, :math:`H` and :math:`W` denote the height and width of those images, respectively.
Then, it outputs a fixed-sized vector for image classification tasks, or an image for a pixel labeling task, etc.
The shape of the output is varied for each task setting.

Let's start from defining LeNet5 in Chainer. It comprises 7 layers.

.. testcode::

    class ConvNet(Chain):
        def __init__(self):
            super(ConvNet, self).__init__(
                conv1=L.Convolution2D(6, ksize=5, stride=1, pad=2),
                conv2=L.Convolution2D(6, ksize=5, stride=1, pad=2),
