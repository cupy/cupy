# DCGAN

This is an example implementation of DCGAN (https://arxiv.org/abs/1511.06434) using trainer.

This code uses Cifar-10 dataset by default.
You can use your own dataset by specifying `--dataset` argument to the directory consists of image files for training.
The model assumes the resolution of an input image is 32x32.
If you want to use another image resolution, you need to change the network architecture in net.py.
