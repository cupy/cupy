import os
import tempfile

import numpy
from six.moves.urillib import request


_dataset_root = os.environ.get('CHAINER_DATASET_ROOT',
                               os.path.expanduser('~/.chainer/dataset'))


def get_dataset_root():
    """Gets the path to the root directory to download and cache datasets.

    Returns:
        str: The path to the dataset root directory.

    """
    return _dataset_root


def set_dataset_root(path):
    """Sets the root directory to download and cache datasets.

    There are two ways to set the dataset root directory. One is by setting the
    environmant variable ``CHAINER_DATASET_ROOT``. The other is by using this
    function. The default dataset root is ``$HOME/.chainer/dataset``.

    Args:
        path (str): Path to the new dataset root directory.

    """
    global _dataset_root
    _dataset_root = path


def get_dataset_directory(dataset_name, create_directory=True):
    """Gets the path to the directory of given dataset.

    The generated path is just a concatenation of the global root directory
    (see :func:`set_datasets_root` for how to change it) and the dataset name.
    The dataset name can contain slashes, which are treated as path separators.

    Args:
        dataset_name (str): Name of the dataset.
        create_directory (bool): If True (default), this function also creates
            the directory at the first time. If the directory already exists,
            then this option is ignored.

    Returns:
        str: Path to the dataset directory.

    """
    path = os.path.join(_dataset_root, dataset_name)
    if create_directory:
        try:
            os.makedirs(path)
        except OSError:
            pass
    return path


def retrieve(urls, path, converter):
    """Downloads a dataset and saves converted one if needed.

    This function donwloads the dataset files and converts them to a (set of)
    NumPy array(s). The converter function takes a (list of) path(s) to
    downloaded temporary files, and returns a dict of NumPy arrays. The dict of
    arrays is saved to the given path.

    If a file already exists at the path, this function just loads it.

    Args:
        urls (str or list of strs): URLs to download files.
        path (str): Path to which the converted file is saved.
        converter: Function to convert the donwloaded files. It takes a string
            if ``urls`` is a string, or a list of strings if ``urls`` is also a
            list of strings.

    Returns:
        A dictionary or a dictionary-like object.

    """
    if os.path.exists(path):
        return numpy.load(path)
        
    if isinstance(urls, str):
        filenames, _ = request.urlretrieve(urls)
    else:
        filenames = [request.urlretrieve(url)[0] for url in urls]

    result = converter(filenames)

    dirname = os.path.dirname(path)
    tmp_path = tempfile.mkstemp(suffix='.npz', dir=dirname)
    numpy.savez_compressed(tmp_path, **result)

    try:
        os.rename(tmp_path, path)
    except OSError:
        os.remove(tmp_path)
        if not os.path.exists(path):
            raise

    return result
