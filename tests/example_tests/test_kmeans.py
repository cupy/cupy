import os
import shutil
import tempfile
import unittest

from cupy import testing

import example_test


@testing.with_requires('matplotlib')
class TestKmeans(unittest.TestCase):

    def test_default(self):
        output = example_test.run_example(
            'kmeans/kmeans.py', '-m', '1', '--num', '10')
        self.assertRegexpMatches(
            output.decode('utf-8'), r''' CPU :  [0-9\.]+ sec
 GPU :  [0-9\.]+ sec
''')

    def test_custom_kernel(self):
        output = example_test.run_example(
            'kmeans/kmeans.py', '-m', '1', '--num', '10',
            '--use-custom-kernel')
        self.assertRegexpMatches(
            output.decode('utf-8'), r''' CPU :  [0-9\.]+ sec
 GPU :  [0-9\.]+ sec
''')

    def test_result_image(self):
        dir_path = tempfile.mkdtemp()
        try:
            image_path = os.path.join(dir_path, 'kmeans.png')
            example_test.run_example(
                'kmeans/kmeans.py', '-m', '1', '--num', '10', '-o', image_path)
            self.assertTrue(os.path.exists(image_path))
        finally:
            shutil.rmtree(dir_path, ignore_errors=True)
