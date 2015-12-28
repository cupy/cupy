import numpy

from chainer import cuda
from chainer import serializer


class DictionarySerializer(serializer.Serializer):

    """Serializer for dictionary.

    This is the standard serializer in Chainer. The chain hierarchy is simply
    mapped to dictionary key.

    Args:
        dict (dict): The dict that this serializer represents.
        path (str): The base path.

    """
    def __init__(self, dict, path=''):
        self.dict = dict
        self.path = path

    def __getitem__(self, key):
        return DictionarySerializer(self.dict, self.path + key + '/')

    def __call__(self, key, value):
        ret = value
        if isinstance(value, cuda.ndarray):
            value = cuda.to_cpu(value)
        arr = numpy.asarray(value)
        self.dict[self.path + key] = arr
        return ret


def save_npz(filename, obj):
    """Saves an object to the file in NPZ format.

    This is a short-cut function to save only one object into an NPZ file.

    Args:
        filename (str): Target file name.
        obj: Object to be serialized. It must support serialization protocol.

    """
    d = {}
    s = DictionarySerializer(d, '')
    s.save(obj)
    with open(filename, 'w') as f:
        numpy.savez(f, **d)


class NPZDeserializer(serializer.Deserializer):

    """Deserializer for NPZ format.

    This is the standard deserializer in Chainer. This deserializer can be used
    to read an object serialized by :class:`NPZSerializer`.

    Args:
        npz: `npz` file opbject.
        path: The base path that the deserialization starts from.

    """
    def __init__(self, npz, path=''):
        self.npz = npz
        self.path = path

    def __getitem__(self, key):
        return NPZDeserializer(self.npz, self.path + key + '/')

    def __call__(self, key, value):
        print(self.path, self.npz.files)
        dataset = self.npz[self.path + key]
        if isinstance(value, numpy.ndarray):
            numpy.copyto(value, dataset)
        elif isinstance(value, cuda.ndarray):
            value.set(numpy.asarray(dataset))
        else:
            value = type(value)(numpy.asarray(dataset))
        return value


def load_npz(filename, obj):
    """Loads an object from the file in NPZ format.

    This is a short-cut function to load from an `.npz` file that contains only
    one object.

    Args:
        filename (str): Name of the file to be loaded.
        obj: Object to be deserialized. It must support serialization protocol.

    """
    with numpy.load(filename) as f:
        d = NPZDeserializer(f, '')
        d.load(obj)
