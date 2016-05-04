from chainer.dataset import dataset_mixin
from chainer.dataset import download
from chainer.dataset import iterator


DatasetMixin = dataset_mixin.DatasetMixin
Iterator = iterator.Iterator

get_dataset_root = download.get_dataset_root
set_dataset_root = download.set_dataset_root
get_dataset_directory = download.get_dataset_directory
retrieve = download.retrieve
