import glob
import os
import unittest

from chainer import testing


class TestRunnable(unittest.TestCase):

    def test_runnable(self):
        cwd = os.path.dirname(__file__)
        for path in glob.iglob(os.path.join(cwd, '**', '*.py')):
            with open(path) as f:
                source = f.read()
            self.assertIn('testing.run_module(__name__, __file__)',
                          source,
                          '''{0} is not runnable.
Call testing.run_module at the end of the test.'''.format(path))


testing.run_module(__name__, __file__)
