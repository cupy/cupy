import inspect
import sys


def make_decorator(test_case_generator):
    # `test_case_generator` is a callable that receives the source test class
    # (typically a subclass of unittest.TestCase) and returns an iterable of
    # generated test cases.
    # Each element of the iterable is a 3-element tuple:
    # [0] Generated class name
    # [1] Dict of members
    # [2] Method generator
    # The method generator is also a callable that receives an original test
    # method and returns a new test method.

    def f(cls):
        # The input is a bare test case
        module_name = cls.__module__

        # Generate parameterized test cases out of the input test case.
        module = sys.modules[module_name]
        assert module.__name__ == module_name
        for cls_name, members, method_generator in test_case_generator(cls):
            _generate_case(
                cls, module, cls_name, members, method_generator)

        # Remove original base class
        return None
    return f


def _generate_case(base, module, cls_name, mb, method_generator):
    members = mb.copy()
    base_methods = inspect.getmembers(base, predicate=inspect.isfunction)
    for name, value in base_methods:
        if not name.startswith('test_'):
            continue
        value = method_generator(value)
        # If the return value of method_generator is None, None is assigned
        # as the value of the test method and it is ignored by pytest.
        members[name] = value

    cls = type(cls_name, (base,), members)

    # Add new test class to module
    cls.__module__ = module.__name__
    setattr(module, cls_name, cls)
