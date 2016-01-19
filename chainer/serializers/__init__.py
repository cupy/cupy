from chainer.serializers import hdf5
from chainer.serializers import npz

HDF5Serializer = hdf5.HDF5Serializer
HDF5Deserializer = hdf5.HDF5Deserializer
save_hdf5 = hdf5.save_hdf5
load_hdf5 = hdf5.load_hdf5

DictionarySerializer = npz.DictionarySerializer
NpzDeserializer = npz.NpzDeserializer
save_npz = npz.save_npz
load_npz = npz.load_npz
