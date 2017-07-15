import os
import shutil
import tempfile
import unittest

from cupy import testing

import example_test


@testing.with_requires('matplotlib')
class TestGMM(unittest.TestCase):

    def test_gmm(self):
        output = example_test.run_example('gmm/gmm.py', '--num', '10')
        self.assertRegexpMatches(
            output.decode('utf-8'), r'''Running CPU\.\.\.
train_accuracy : [0-9\.]+
test_accuracy : [0-9\.]+
 CPU :  [0-9\.]+ sec
Running GPU\.\.\.
train_accuracy : [0-9\.]+
test_accuracy : [0-9\.]+
 GPU :  [0-9\.]+ sec
''')

    def test_output_image(self):
        dir_path = tempfile.mkdtemp()
        try:
            image_path = os.path.join(dir_path, 'gmm.png')
            output = example_test.run_example(
                'gmm/gmm.py', '--num', '10', '-o', image_path)
            self.assertTrue(os.path.exists(image_path))
        finally:
            shutil.rmtree(dir_path, ignore_errors=True)
