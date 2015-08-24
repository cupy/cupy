
C_CONTIGUOUS = 1
F_CONTIGUOUS = 1 << 1
OWNDATA = 1 << 2

C_DIRTY = 1 << 3  # internal use only
F_DIRTY = 1 << 4  # internal use only


class Flags(object):

    __slots__ = ['_value']

    def __init__(self, value):
        self._value = value

    def __getitem__(self, name):
        if name == 'C_CONTIGUOUS':
            return self.c_contiguous
        elif name == 'F_CONTIGUOUS':
            return self.f_contiguous
        elif name == 'OWNDATA':
            return self.owndata
        else:
            raise KeyError('%s is not defined for cupy.ndarray.flags' % name)

    def __repr__(self):
        t = '  %s : %s'
        ret = []
        for name in 'C_CONTIGUOUS', 'F_CONTIGUOUS', 'OWNDATA':
            ret.append(t % (name, self[name]))
        return '\n'.join(ret)

    @property
    def c_contiguous(self):
        return bool(self._value & C_CONTIGUOUS)

    @property
    def f_contiguous(self):
        return bool(self._value & F_CONTIGUOUS)

    @property
    def owndata(self):
        return bool(self._value & OWNDATA)
