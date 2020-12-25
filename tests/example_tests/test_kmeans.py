import os
import re
import shutil
import tempfile
import unittest

from cupy import testing

from example_tests import example_test


os.environ['MPLBACKEND'] = 'Agg'


@testing.with_requires('matplotlib')
class TestKmeans(unittest.TestCase):

    def test_default(self):
        output = example_test.run_example(
            'kmeans/kmeans.py', '-m', '1', '--num', '10')
        assert re.search(
            r' CPU :  [0-9\.]+ sec\s+GPU :  [0-9\.]+ sec',
            output.decode('utf-8'),
        )

    def test_custom_kernel(self):
        output = example_test.run_example(
            'kmeans/kmeans.py', '-m', '1', '--num', '10',
            '--use-custom-kernel')
        assert re.search(
            r' CPU :  [0-9\.]+ sec\s+GPU :  [0-9\.]+ sec',
            output.decode('utf-8'),
        )

    def test_result_image(self):
        dir_path = tempfile.mkdtemp()
        try:
            image_path = os.path.join(dir_path, 'kmeans.png')
            example_test.run_example(
                'kmeans/kmeans.py', '-m', '1', '--num', '10', '-o', image_path)
            assert os.path.exists(image_path)
        finally:
            shutil.rmtree(dir_path, ignore_errors=True)
