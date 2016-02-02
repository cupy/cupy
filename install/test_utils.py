import unittest

import utils


class TestPrintWarning(unittest.TestCase):

    def test_print_warning(self):
        utils.print_warning('This is a test.')
