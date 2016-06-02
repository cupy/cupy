import os

import numpy

from chainer.dataset import download


def get_ptb_words():
    """Gets the Penn Tree Bank dataset as long word sequences.

    `Penn Tree Bank <https://www.cis.upenn.edu/~treebank/>`_ is originally a
    corpus of English sentences with linguistic structure annotations. This
    function uses a variant distributed at
    `https://github.com/tomsercu/lstm <https://github.com/tomsercu/lstm>`_,
    which omits the annotation and splits the dataset into three parts:
    training, validation, and test.

    This function returns the training, validation, and test sets, each of
    which is represented as a long array of word IDs. All sentences in the
    dataset are concatenated by End-of-Sentence mark '<eos>', which is treated
    as one of the vocabulary.

    Returns:
        tuple of numpy.ndarray: Int32 vectors of word IDs.

    .. Seealso::
       Use :func:`get_ptb_words_vocabulary` to get the mapping between the
       words and word IDs.

    """
    train = _retrieve_ptb_words('train.npz', _train_url)
    valid = _retrieve_ptb_words('valid.npz', _valid_url)
    test = _retrieve_ptb_words('test.npz', _test_url)
    return train, valid, test


def get_ptb_words_vocabulary():
    """Gets the Penn Tree Bank word vocabulary.

    Returns:
        dict: Dictionary that maps words to corresponding word IDs. The IDs are
            used in the Penn Tree Bank long sequence datasets.

    .. seealso::
       See :func:`get_ptb_words` for the actual datasets.

    """
    return _retrieve_word_vocabulary()


_train_url = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt'  # NOQA
_valid_url = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt'  # NOQA
_test_url = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt'  # NOQA


def _retrieve_ptb_words(name, url):
    def creator(path):
        vocab = _retrieve_word_vocabulary()
        words = _load_words(url)
        x = numpy.empty(len(words), dtype=numpy.int32)
        for i, word in enumerate(words):
            x[i] = vocab[word]

        numpy.savez_compressed(path, x=x)
        return {'x': x}

    root = download.get_dataset_directory('pfnet/chainer/ptb')
    path = os.path.join(root, name)
    loaded = download.cache_or_load_file(path, creator, numpy.load)
    return loaded['x']


def _retrieve_word_vocabulary():
    def creator(path):
        words = _load_words(_train_url)
        vocab = {}
        index = 0
        with open(path, 'w') as f:
            for word in words:
                if word not in vocab:
                    vocab[word] = index
                    index += 1
                    f.write(word + '\n')

        return vocab

    def loader(path):
        vocab = {}
        with open(path) as f:
            for i, word in enumerate(f):
                vocab[word.strip()] = i
        return vocab

    root = download.get_dataset_directory('pfnet/chainer/ptb')
    path = os.path.join(root, 'vocab.txt')
    return download.cache_or_load_file(path, creator, loader)


def _load_words(url):
    path = download.cached_download(url)
    words = []
    with open(path) as words_file:
        for line in words_file:
            if line:
                words += line.strip().split()
                words.append('<eos>')
    return words
