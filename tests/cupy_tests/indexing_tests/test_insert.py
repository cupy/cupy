import unittest

from cupy import testing


@testing.gpu
class TestInsert(unittest.TestCase):

    _multiprocess_can_split_ = True
