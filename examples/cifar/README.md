# Convolutional neural networks for CIFAR-10 and CIFAR-100 Classification

This is an example of a convolutional neural network (convnet) applied to an image classification task using the CIFAR-10 or CIFAR-100 dataset. The CIFAR datasets can be a good choice for initial experiments with convnets because the size and number of images is small enough to allow typical models to be trained in a reasonable amount of time. However, the classification task is still challenging because natural images are used.

Specifically, there are 50000 color training images of size 32x32 pixels with either 10 class labels (for CIFAR-10) or 100 class labels (for CIFAR-100).

For CIFAR-10, state of the art methods without data augmentation can achieve similar to human-level classification accuracy of around 94%.
For CIFAR-100, state of the art without data augmentation is around 20% (DenseNet).

The code consists of three parts: dataset preparation, network and optimizer definition and learning loop, similar to the MNIST example.

More models may be added in the future, but currently only one model is available that uses the VGG-style network from [here](http://torch.ch/blog/2015/07/30/cifar.html) which is based on the network architecture from the paper from [here](http://arxiv.org/pdf/1409.1556v6.pdf).

No data augmentation is used and the classification accuracy on the CIFAR-10 test set for the VGG-style model should reach approximately 89% after 200 iterations or so.

If you want to run this example on the N-th GPU, pass `--gpu=N` to the script. To run on CPU, pass `--gpu=-1`.

For example, to run the default model, which uses CIFAR-10 and GPU 0:
```
train_cifar.py
```

to run the CIFAR-100 dataset on GPU 1:
```
train_cifar.py --gpu=1 --dataset='cifar100'
```
