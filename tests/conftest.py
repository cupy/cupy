import os
import subprocess
import sys

from chainer import testing
from chainer.testing import parameterized


_pairwise_parameterize = (
    os.environ.get('CHAINER_TEST_PAIRWISE_PARAMETERIZATION', 'never'))
assert _pairwise_parameterize in ('never', 'always')


def _is_pip_installed():
    try:
        import pip  # NOQA
        return True
    except ImportError:
        return False


def _is_in_ci():
    ci_name = os.environ.get('CHAINER_CI', '')
    return ci_name != ''


def pytest_configure(config):
    # Print installed packages
    if _is_in_ci() and _is_pip_installed():
        print("***** Installed packages *****", flush=True)
        subprocess.check_call([sys.executable, '-m', 'pip', 'freeze', '--all'])


def pytest_collection(session):
    # Perform pairwise testing.
    # TODO(kataoka): This is a tentative fix. Discuss its public interface.
    if _pairwise_parameterize == 'always':
        pairwise_product_dict = parameterized._pairwise_product_dict
        testing.product_dict = pairwise_product_dict
        parameterized.product_dict = pairwise_product_dict


def pytest_collection_finish(session):
    if _pairwise_parameterize == 'always':
        product_dict = parameterized._product_dict_orig
        testing.product_dict = product_dict
        parameterized.product_dict = product_dict
