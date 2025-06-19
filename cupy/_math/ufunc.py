from cupy import _core


def create_math_ufunc(math_name, nargs, name, doc, support_complex=True):
    assert 1 <= nargs <= 2
    if nargs == 1:
        types = ('e->e', 'f->f', 'd->d')
        if support_complex:
            types += ('F->F', 'D->D')
        return _core.create_ufunc(
            name, types, 'out0 = %s(in0)' % math_name, doc=doc)
    else:
        types = ('ee->e', 'ff->f', 'dd->d')
        if support_complex:
            types += ('FF->F', 'DD->D')
        return _core.create_ufunc(
            name, types, 'out0 = %s(in0, in1)' % math_name, doc=doc)
