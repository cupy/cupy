from cupy.cuda import driver


class Event(object):

    """CUDA event."""

    def __init__(self, block=False, disable_timing=False, interprocess=False):
        flag = 0
        if block:
            flag |= 1
        if disable_timing:
            flag |= 2
        if interprocess:
            flag |= 4
        self.ptr = driver.eventCreate(flag)

    def __del__(self):
        driver.eventDestroy(self.ptr)

    def synchronize(self):
        driver.eventSynchronize(self.ptr)


class Stream(object):

    """CUDA stream."""

    def __init__(self, null=False):
        if null:
            self.ptr = driver.Stream()
        else:
            self.ptr = driver.streamCreate(0)

    def __del__(self):
        if self.ptr:
            driver.streamDestroy(self.ptr)
            self.ptr = driver.Stream()

    def synchronize(self):
        driver.streamSynchronize(self.ptr)

    def add_callback(self, callback, arg):
        driver.streamAddCallback(self.ptr, callback, arg)

    def record(self, event=None):
        if event is None:
            event = Event()
        driver.eventRecord(event.ptr, self.ptr)
        return event
