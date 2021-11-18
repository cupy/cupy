import unittest

import pytest

from cupy.cuda import nccl
from cupy.testing import _condition
from cupyx.distributed import _store


nccl_available = nccl.available


@pytest.mark.skipif(not nccl_available, reason='nccl is not installed')
class TestTCPStore(unittest.TestCase):

    @_condition.retry(10)
    def test_store_get_set(self):
        store = _store.TCPStore(1)
        store.run()
        try:
            proxy = _store.TCPStoreProxy()
            proxy['test-value'] = 1234
            assert proxy['test-value'] == 1234
            proxy['test-bytes'] = b'123\x00123'
            assert proxy['test-bytes'] == b'123\x00123'
        finally:
            store.stop()

    @_condition.retry(10)
    def test_store_invalid_get_set(self):
        store = _store.TCPStore(1)
        try:
            store.run()
            proxy = _store.TCPStoreProxy()
            with pytest.raises(ValueError):
                proxy['test-value'] = 1234.0
            with pytest.raises(ValueError):
                proxy[123] = 1234
            with pytest.raises(ValueError):
                a = proxy[123]  # NOQA
        finally:
            store.stop()
    # Barrier is tested directly in the communicators
