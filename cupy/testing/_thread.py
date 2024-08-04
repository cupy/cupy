import threading


class TestThread(threading.Thread):
    """A thread that can propagate exceptions.

    This class can be used as a drop-in replacement to `threading.Thread`.
    Exception raised in the thread will be reraised upon `join()`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exc = None

    def run(self, *args, **kwargs):
        try:
            super().run(*args, **kwargs)
        except Exception as e:
            self._exc = e

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        if self._exc is not None:
            raise self._exc
