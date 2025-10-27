import cupy

def test_ipc_handle_creation():
    a = cupy.arange(10)
    handle = cupy.get_ipc_handle(a)
    assert handle is not None
