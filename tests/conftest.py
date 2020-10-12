import os
import subprocess
import sys


def _is_pip_installed():
    try:
        import pip  # NOQA
        return True
    except ImportError:
        return False


def _is_in_ci():
    ci_name = os.environ.get('CUPY_CI', '')
    return ci_name != ''


def pytest_configure(config):
    # Print installed packages
    if _is_in_ci() and _is_pip_installed():
        print("***** Installed packages *****", flush=True)
        subprocess.check_call([sys.executable, '-m', 'pip', 'freeze', '--all'])
