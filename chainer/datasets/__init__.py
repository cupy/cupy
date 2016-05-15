from chainer.datasets import dict_dataset
from chainer.datasets import image_dataset
from chainer.datasets import sub_dataset
from chainer.datasets import tuple_dataset


DictDataset = dict_dataset.DictDataset
ImageDataset = image_dataset.ImageDataset
SubDataset = sub_dataset.SubDataset
TupleDataset = tuple_dataset.TupleDataset

get_cross_validation_datasets = sub_dataset.get_cross_validation_datasets
get_cross_validation_datasets_random = (
    sub_dataset.get_cross_validation_datasets_random)

split_dataset = sub_dataset.split_dataset
split_dataset_random = sub_dataset.split_dataset_random
