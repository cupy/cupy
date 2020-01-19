import collections
import inspect
import sys


# A tuple that represents a test case.
# For bare (non-generated) test cases, [1] and [2] are None.
# [0] Test case class
# [1] Module name in whicn the class is defined
# [2] Class name
_TestCaseTuple = collections.namedtuple(
    '_TestCaseTuple', ('klass', 'module_name', 'class_name'))


class _ParameterizedTestCaseBundle(object):
    def __init__(self, cases):
        # cases is a list of _TestCaseTuple's
        assert isinstance(cases, list)
        assert all(isinstance(tup, _TestCaseTuple) for tup in cases)

        self.cases = cases


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

    def f(cases):
        if isinstance(cases, _ParameterizedTestCaseBundle):
            # The input is a parameterized test case.
            cases = cases.cases
        else:
            # Input is a bare test case, i.e. not one generated from another
            # parameterize.
            cases = [_TestCaseTuple(cases, None, None)]

        generated_cases = []
        for klass, mod_name, cls_name in cases:
            if mod_name is not None:
                # The input is a parameterized test case.
                # Remove it from its module.
                delattr(sys.modules[mod_name], cls_name)
            else:
                # The input is a bare test case
                mod_name = klass.__module__

            # Generate parameterized test cases out of the input test case.
            c = _generate_test_cases(mod_name, klass, test_case_generator)
            generated_cases += c

        # Return the bundle of generated cases to allow repeated application of
        # parameterize decorators.
        return _ParameterizedTestCaseBundle(generated_cases)
    return f


def _generate_case(base, module, cls_name, mb, method_generator):
    # Returns a _TestCaseTuple.

    members = mb.copy()
    # ismethod for Python 2 and isfunction for Python 3
    base_methods = inspect.getmembers(
        base, predicate=lambda m: inspect.ismethod(m) or inspect.isfunction(m))
    for name, value in base_methods:
        if not name.startswith('test_'):
            continue
        value = method_generator(value)
        # If the return value of method_generator is None, None is assigned
        # as the value of the test method and it is ignored by pytest.
        members[name] = value

    cls = type(cls_name, (base,), members)

    # Add new test class to module
    setattr(module, cls_name, cls)

    return _TestCaseTuple(cls, module.__name__, cls_name)


def _generate_test_cases(module_name, base_class, test_case_generator):
    # Returns a list of _TestCaseTuple's holding generated test cases.
    module = sys.modules[module_name]

    generated_cases = []
    for cls_name, members, method_generator in (
            test_case_generator(base_class)):
        c = _generate_case(
            base_class, module, cls_name, members, method_generator)
        generated_cases.append(c)

    return generated_cases
