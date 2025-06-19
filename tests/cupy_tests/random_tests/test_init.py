import cupy


def test_bytes():
    out = cupy.random.bytes(10)
    assert isinstance(out, bytes)
