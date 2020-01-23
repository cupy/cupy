import io
import pickle
import unittest

import cupy
from cupy import testing


@testing.gpu
class TestNpz(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_save_load(self, dtype):
        a = testing.shaped_arange((2, 3, 4), dtype=dtype)
        sio = io.BytesIO()
        cupy.save(sio, a)
        s = sio.getvalue()
        sio.close()

        sio = io.BytesIO(s)
        b = cupy.load(sio)
        sio.close()

        testing.assert_array_equal(a, b)

    def test_save_pickle(self):
        data = object()

        sio = io.BytesIO()
        with self.assertRaises(ValueError):
            cupy.save(sio, data, allow_pickle=False)
        sio.close()

        sio = io.BytesIO()
        cupy.save(sio, data, allow_pickle=True)
        sio.close()

    def test_load_pickle(self):
        a = testing.shaped_arange((2, 3, 4), dtype=cupy.float32)

        sio = io.BytesIO()
        a.dump(sio)
        s = sio.getvalue()
        sio.close()

        sio = io.BytesIO(s)
        b = cupy.load(sio, allow_pickle=True)
        testing.assert_array_equal(a, b)
        sio.close()

        sio = io.BytesIO(s)
        with self.assertRaises(ValueError):
            cupy.load(sio, allow_pickle=False)
        sio.close()

    @testing.for_all_dtypes()
    def check_savez(self, savez, dtype):
        a1 = testing.shaped_arange((2, 3, 4), dtype=dtype)
        a2 = testing.shaped_arange((3, 4, 5), dtype=dtype)

        sio = io.BytesIO()
        savez(sio, a1, a2)
        s = sio.getvalue()
        sio.close()

        sio = io.BytesIO(s)
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
        s = pickle.dumps(a)
        b = pickle.loads(s)
        testing.assert_array_equal(a, b)

    @testing.for_all_dtypes()
    def test_dump(self, dtype):
        a = testing.shaped_arange((2, 3, 4), dtype=dtype)

        sio = io.BytesIO()
        a.dump(sio)
        s = sio.getvalue()
        sio.close()

        sio = io.BytesIO(s)
        b = cupy.load(sio, allow_pickle=True)
        sio.close()

        testing.assert_array_equal(a, b)
