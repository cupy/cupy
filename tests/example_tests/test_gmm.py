import os
import shutil
import tempfile
import unittest

from cupy import testing

from example_tests import example_test


os.environ['MPLBACKEND'] = 'Agg'


@testing.with_requires('matplotlib')
@testing.with_requires('scipy')
class TestGMM(unittest.TestCase):

    def test_gmm(self):
        output = example_test.run_example('gmm/gmm.py', '--num', '10')
        self.assertRegex(
            output.decode('utf-8'),
            r'Running CPU\.\.\.\s+train_accuracy : [0-9\.]+\s+' +
            r'test_accuracy : [0-9\.]+\s+CPU :  [0-9\.]+ sec\s+' +
            r'Running GPU\.\.\.\s+train_accuracy : [0-9\.]+\s+' +
            r'test_accuracy : [0-9\.]+\s+GPU :  [0-9\.]+ sec')

    def test_output_image(self):
        dir_path = tempfile.mkdtemp()
        try:
            image_path = os.path.join(dir_path, 'gmm.png')
            example_test.run_example(
                'gmm/gmm.py', '--num', '10', '-o', image_path)
            self.assertTrue(os.path.exists(image_path))
        finally:
            shutil.rmtree(dir_path, ignore_errors=True)
