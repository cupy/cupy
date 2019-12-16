import numpy


def is_indexing_cause_synchronize(indexes):
    # Returns whether advanced indexing with the given indexes causes
    # synchronization.
    if isinstance(indexes, bool):
        return True
    if (isinstance(indexes, numpy.ndarray)
            and indexes.dtype == numpy.bool_):
        return True
    if (isinstance(indexes, tuple)
            and any(is_indexing_cause_synchronize(idx)
                    for idx in indexes)):
        return True
    return False
