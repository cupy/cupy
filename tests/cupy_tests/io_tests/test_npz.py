import unittest

import six

import cupy
from cupy import testing


@testing.gpu
class TestNpz(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    def test_save_load(self, dtype):
        a = testing.shaped_arange((2, 3, 4), dtype=dtype)
        sio = six.BytesIO()
        cupy.save(sio, a)
        s = sio.getvalue()
        sio.close()

        sio = six.BytesIO(s)
        b = cupy.load(sio)
        sio.close()

        testing.assert_array_equal(a, b)

    @testing.for_all_dtypes()
    def check_savez(self, savez, dtype):
        a1 = testing.shaped_arange((2, 3, 4), dtype=dtype)
        a2 = testing.shaped_arange((3, 4, 5), dtype=dtype)

        sio = six.BytesIO()
        savez(sio, a1, a2)
        s = sio.getvalue()
        sio.close()

        sio = six.BytesIO(s)
        with cupy.load(sio) as d:
            b1 = d['arr_0']
            b2 = d['arr_1']
        sio.close()

        testing.assert_array_equal(a1, b1)
        testing.assert_array_equal(a2, b2)

    def test_savez(self):
        self.check_savez(cupy.savez)

    def test_savez_compressed(self):
        self.check_savez(cupy.savez_compressed)

    @testing.for_all_dtypes()
    def test_pickle(self, dtype):
        a = testing.shaped_arange((2, 3, 4), dtype=dtype)
        s = six.moves.cPickle.dumps(a)
        b = six.moves.cPickle.loads(s)
        testing.assert_array_equal(a, b)

    @testing.for_all_dtypes()
    def test_dump(self, dtype):
        a = testing.shaped_arange((2, 3, 4), dtype=dtype)

        sio = six.BytesIO()
        a.dump(sio)
        s = sio.getvalue()
        sio.close()

        sio = six.BytesIO(s)
        b = cupy.load(sio)
        sio.close()

        testing.assert_array_equal(a, b)
