.. module:: chainer.serializers

Serializers
===========

Serialization in NumPy NPZ format
---------------------------------

NumPy serializers can be used in arbitrary environments that Chainer runs with.
It consists of asymmetric serializer/deserializer due to the fact that :func:`numpy.savez` does not support online serialization.
Therefore, serialization requires two-step manipulation: first packing the objects into a flat dictionary, and then serializing it into npz format.

.. autoclass:: DictionarySerializer
.. autoclass:: NpzDeserializer
.. autofunction:: save_npz
.. autofunction:: load_npz

Serialization in HDF5 format
----------------------------
.. autoclass:: HDF5Serializer
.. autoclass:: HDF5Deserializer
.. autofunction:: save_hdf5
.. autofunction:: load_hdf5
