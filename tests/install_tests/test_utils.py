import unittest

from install import utils


class TestPrintWarning(unittest.TestCase):

    def test_print_warning(self):
        utils.print_warning('This is a test.')


class TestSearchOnPath(unittest.TestCase):

    def test_exec_not_found(self):
        self.assertIsNone(utils.search_on_path(['no_such_exec']))
