from cupy import reduction

argmax = reduction.argmax


def nanargmax(a, axis=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


argmin = reduction.argmin


def nanargmin(a, axis=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def argwhere(a, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def nonzero(a, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def flatnonzero(a, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def where(condition, x=None, y=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def searchsorted(a, v, side='left', sorter=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def extract(condition, arr, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError
