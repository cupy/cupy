import nose

from chainer.testing import helper
from chainer.testing import parameterized

parameterize = parameterized.parameterize
product = parameterized.product

with_requires = helper.with_requires


def run_module(name, file):
    """Run current test cases of the file.

    Args:
        name: __name__ attrinute of the file.
        file: __file__ attribute of the file.
    """

    if name == '__main__':

        nose.runmodule(argv=[file, '-vvs', '-x', '--pdb', '--pdb-failure'],
                       exit=False)
