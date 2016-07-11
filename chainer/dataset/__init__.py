from chainer.dataset import convert
from chainer.dataset import dataset_mixin
from chainer.dataset import download
from chainer.dataset import iterator


DatasetMixin = dataset_mixin.DatasetMixin
Iterator = iterator.Iterator

concat_examples = convert.concat_examples

get_dataset_root = download.get_dataset_root
set_dataset_root = download.set_dataset_root
get_dataset_directory = download.get_dataset_directory
cached_download = download.cached_download
cache_or_load_file = download.cache_or_load_file
