import pytest

from cupyx.distributed import _store


class TestTCPStore:

    def test_store_get_set(self):
        store = _store.TCPStore(1)
        store.run()
        proxy = _store.TCPStoreProxy()
        proxy['test-value'] = 1234
        assert proxy['test-value'] == 1234
        proxy['test-bytes'] = b'123\x00123'
        assert proxy['test-bytes'] == b'123\x00123'

    def test_store_invalid_get_set(self):
        store = _store.TCPStore(1)
        store.run()
        proxy = _store.TCPStoreProxy()
        with pytest.raises(ValueError):
            proxy['test-value'] = 1234.0
        with pytest.raises(ValueError):
            proxy[123] = 1234
        with pytest.raises(ValueError):
            a = proxy[123]  # NOQA
    # Barrier is tested directly in the communicators
