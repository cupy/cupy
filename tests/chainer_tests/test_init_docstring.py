import importlib
import inspect
import pkg_resources
import pkgutil
import unittest

import chainer
from chainer import testing


def get_init_doc(klass):
    for attr in inspect.classify_class_attrs(klass):
        if attr.name == '__init__':
            if attr.defining_class is klass:
                return attr.object.__doc__
            else:
                # Ignore __init__ method inherited from a super class
                return None
    return None


def get_protobuf_version():
    ws = pkg_resources.WorkingSet()
    try:
        ws.require('protobuf<3.0.0')
        return 2
    except pkg_resources.VersionConflict:
        return 3


class TestInitDocstring(unittest.TestCase):

    def check_init_docstring(self, mod, errors):
        for name, value in inspect.getmembers(mod):
            if not inspect.isclass(value):
                continue
            init_doc = get_init_doc(value)
            if init_doc == object.__init__.__doc__:
                # Ignore doc string inherited from `object`
                continue

            if init_doc is not None:
                # Do not permit to write docstring in `__init__`
                errors.append((mod, value, init_doc))

    def test_init_docstring_empty(self):
        protobuf_ver = get_protobuf_version()
        errors = []
        root = chainer.__file__
        for _, modname, _ in pkgutil.walk_packages(root):
            if 'chainer' not in modname:
                # Skip tests
                continue

            if protobuf_ver != 2 and 'protobuf2':
                continue
            if protobuf_ver != 3 and 'protobuf3':
                continue

            try:
                # imoprter of pkgutil causes an error.
                # We use importlib to import modules directly.
                mod = importlib.import_module(modname)
            except Exception as e:
                print('Failed to load module accidentally: %s' %
                      modname)
                print(e)
                print('')
                continue

            self.check_init_docstring(mod, errors)

        if errors:
            msg = ''
            for mod, value, init_doc in errors:
                msg += '{}.{} has __init__.__doc__:\n{}\n\n'.format(
                    mod.__name__, value, init_doc)
            self.fail(msg)


testing.run_module(__name__, __file__)
