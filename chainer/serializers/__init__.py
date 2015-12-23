def _unsupported(msg):
    def f(*args, **nargs):
        raise RuntimeError(msg)
    return f


def _unsupported_class(msg):

    class _Unsupported(object):

        def __init__(self, *args, **vargs):
            raise RuntimeError(msg)

    return _Unsupported


try:
    from chainer.serializers import hdf5

    HDF5Serializer = hdf5.HDF5Serializer
    HDF5Deserializer = hdf5.HDF5Deserializer
    save_hdf5 = hdf5.save_hdf5
    load_hdf5 = hdf5.load_hdf5

except ImportError:

    msg = '''h5py is not installed on your environment.
Please install h5py to activate hdf5 serializers.

  $ pip install h5py
    '''
    HDF5Serializer = _unsupported_class(msg)
    HDF5Deserializer = _unsupported_class(msg)
    save_hdf5 = _unsupported(msg)
    load_hdf5 = _unsupported(msg)
