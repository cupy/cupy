import numpy

from chainer import cuda
from chainer import serializer


try:
    import h5py
    _available = True
except ImportError:
    _available = False


def _check_available():
    if not _available:
        msg = '''h5py is not installed on your environment.
Please install h5py to activate hdf5 serializers.

  $ pip install h5py'''
        raise RuntimeError(msg)


class HDF5Serializer(serializer.Serializer):

    """Serializer for HDF5 format.

    This is the standard serializer in Chainer. The chain hierarchy is simply
    mapped to HDF5 hierarchical groups.

    Args:
        group (h5py.Group): The group that this serializer represents.
        compression (int): Gzip compression level.

    """

    def __init__(self, group, compression=4):
        _check_available()

        self.group = group
        self.compression = compression

    def __getitem__(self, key):
        name = self.group.name + '/' + key
        return HDF5Serializer(self.group.require_group(name), self.compression)

    def __call__(self, key, value):
        ret = value
        if isinstance(value, cuda.ndarray):
            value = cuda.to_cpu(value)
        arr = numpy.asarray(value)
        compression = None if arr.size <= 1 else self.compression
        self.group.create_dataset(key, data=arr, compression=compression)
        return ret


def save_hdf5(filename, obj, compression=4):
    """Saves an object to the file in HDF5 format.

    This is a short-cut function to save only one object into an HDF5 file. If
    you want to save multiple objects to one HDF5 file, use
    :class:`HDF5Serializer` directly by passing appropriate :class:`h5py.Group`
    objects.

    Args:
        filename (str): Target file name.
        obj: Object to be serialized. It must support serialization protocol.
        compression (int): Gzip compression level.

    """
    _check_available()
    with h5py.File(filename, 'w') as f:
        s = HDF5Serializer(f, compression=compression)
        s.save(obj)


class HDF5Deserializer(serializer.Deserializer):

    """Deserializer for HDF5 format.

    This is the standard deserializer in Chainer. This deserializer can be used
    to read an object serialized by :class:`HDF5Serializer`.

    Args:
        group (h5py.Group): The group that the deserialization starts from.
        strict (bool): If ``True``, the deserializer raises an error when an
            expected value is not found in the given HDF5 file. Otherwise,
            it ignores the value and skip deserialization.

    """

    def __init__(self, group, strict=True):
        _check_available()
        self.group = group
        self.strict = strict

    def __getitem__(self, key):
        name = self.group.name + '/' + key
        try:
            group = self.group.require_group(name)
        except ValueError:
            # require_group raises ValueError if there does not exist
            # the given group and the file is read mode.
            group = None
        return HDF5Deserializer(group, strict=self.strict)

    def __call__(self, key, value):
        if self.group is None:
            if not self.strict:
                return value
            else:
                raise ValueError('Inexistent group is specified')
        if not self.strict and key not in self.group:
            return value

        dataset = self.group[key]
        if value is None:
            return numpy.asarray(dataset)
        elif isinstance(value, numpy.ndarray):
            dataset.read_direct(value)
        elif isinstance(value, cuda.ndarray):
            value.set(numpy.asarray(dataset))
        else:
            value = type(value)(numpy.asarray(dataset))
        return value


def load_hdf5(filename, obj):
    """Loads an object from the file in HDF5 format.

    This is a short-cut function to load from an HDF5 file that contains only
    one object. If you want to load multiple objects from one HDF5 file, use
    :class:`HDF5Deserializer` directly by passing appropriate
    :class:`h5py.Group` objects.

    Args:
        filename (str): Name of the file to be loaded.
        obj: Object to be deserialized. It must support serialization protocol.

    """
    _check_available()
    with h5py.File(filename, 'r') as f:
        d = HDF5Deserializer(f)
        d.load(obj)
