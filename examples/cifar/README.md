# Convolutional nueral networks for CIFAR Classification

This is a an example of convolutional neural networks (convnets) applied to and image classification task using the CIFAR-10 or CIFAR-100 dataset. The CIFAR datasets can be a good choice for initial prototyping with convnets because the image size and number of images is still small enough that training can be completed in a reasonable amount of time on a single mid-range GPU. Specifically, there are 50000 color training images of size 32x32 pixels with either 10 class labels (for CIFAR-10) or 100 class labels (for CIFAR-100). These datasets are also challenging because natural images are used.

For CIFAR-10, state of the art methods without data augmentation can achieve similar to human-level classification accuracy of around 94%.
For CIFAR-100, state of the art without data augmentation is around 20% (DenseNet) but we are not aware of any estimates of human level accuracy for comparison.

The code consists of three parts: dataset preparation, network and optimizer definition and learning loop.
This is a common routine to write a learning process of networks with dataset that is small enough to fit into memory.

If you want to run this example on the N-th GPU, pass `--gpu=N` to the script.

For example, to run the CIFAR-10 dataset on GPU 0:
```
train_cifar.py --gpu=0 --dataset='cifar10'
```

todo: select between models.
