import importlib
import os
import sys


def _from_install_import(name):
    install_module_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'install')
    original_sys_path = sys.path.copy()
    try:
        sys.path.append(install_module_path)
        return importlib.import_module(name)
    finally:
        sys.path = original_sys_path
