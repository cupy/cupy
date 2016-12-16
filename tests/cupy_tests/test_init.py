import nose
import unittest


class TestImportError(unittest.TestCase):

    def test_import_error(self):
        try:
            import cupy  # noqa
        except Exception as e:
            self.assertIsInstance(e, RuntimeError)

# This is copied from chainer/testing/__init__.py, so should be replaced in
# some way.
if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
