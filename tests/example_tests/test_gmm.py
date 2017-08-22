import os
import shutil
import tempfile
import unittest

import six

from cupy import testing

import example_test


@testing.with_requires('matplotlib')
class TestGMM(unittest.TestCase):

    def test_gmm(self):
        output = example_test.run_example('gmm/gmm.py', '--num', '10')
        six.assertRegex(
            self, output.decode('utf-8'),
            'Running CPU\\.\\.\\.\s+train_accuracy : [0-9\\.]+\s+' +
            'test_accuracy : [0-9\\.]+\s+CPU :  [0-9\\.]+ sec\s+' +
            'Running GPU\\.\\.\\.\s+train_accuracy : [0-9\\.]+\s+' +
            'test_accuracy : [0-9\\.]+\s+GPU :  [0-9\\.]+ sec')

    def test_output_image(self):
        dir_path = tempfile.mkdtemp()
        try:
            image_path = os.path.join(dir_path, 'gmm.png')
            example_test.run_example(
                'gmm/gmm.py', '--num', '10', '-o', image_path)
            self.assertTrue(os.path.exists(image_path))
        finally:
            shutil.rmtree(dir_path, ignore_errors=True)
