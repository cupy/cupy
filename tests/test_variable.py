from unittest import TestCase
import numpy

import chain

class TestVariable(TestCase):
    def test_initialize_by_array(self):
        a = numpy.empty((4, 3))
        x = chain.Variable(a)
        self.assertTrue((a == x.data).all())
