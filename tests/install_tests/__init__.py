import importlib
import os
import sys


source_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))


def _from_install_import(name):
    original_sys_path = sys.path.copy()
    try:
        sys.path.append(os.path.join(source_root, 'install'))
        return importlib.import_module(name)
    finally:
        sys.path = original_sys_path


cupy_builder = _from_install_import('cupy_builder')
cupy_builder.initialize(cupy_builder.Context(source_root, _env={}, _argv=[]))
