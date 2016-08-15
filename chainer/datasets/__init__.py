from chainer.datasets import cifar
from chainer.datasets import dict_dataset
from chainer.datasets import image_dataset
from chainer.datasets import mnist
from chainer.datasets import ptb
from chainer.datasets import sub_dataset
from chainer.datasets import tuple_dataset


DictDataset = dict_dataset.DictDataset
ImageDataset = image_dataset.ImageDataset
LabeledImageDataset = image_dataset.LabeledImageDataset
SubDataset = sub_dataset.SubDataset
TupleDataset = tuple_dataset.TupleDataset

get_cross_validation_datasets = sub_dataset.get_cross_validation_datasets
get_cross_validation_datasets_random = (
    sub_dataset.get_cross_validation_datasets_random)

split_dataset = sub_dataset.split_dataset
split_dataset_random = sub_dataset.split_dataset_random


# examples

get_cifar10 = cifar.get_cifar10
get_cifar100 = cifar.get_cifar100
get_mnist = mnist.get_mnist
get_ptb_words = ptb.get_ptb_words
get_ptb_words_vocabulary = ptb.get_ptb_words_vocabulary
