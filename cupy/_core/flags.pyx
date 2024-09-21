# distutils: language = c++


class Flags(object):

    def __init__(self, array):
        self._array = array
        self._c_contiguous = array._c_contiguous
        self._f_contiguous = array._f_contiguous
        self._owndata = array.base is None
        self._writeable = array._writeable

    @property
    def c_contiguous(self):
        return self._c_contiguous

    @property
    def f_contiguous(self):
        return self._f_contiguous

    @property
    def owndata(self):
        return self._owndata

    @property
    def writeable(self):
        return self._writeable

    @writeable.setter
    def writeable(self, new_writeable):
        base = self._array.base
        if base is not None and not base._writeable and new_writeable:
            raise ValueError('cannot set WRITEABLE flag to True of this array')
        self._writeable = new_writeable
        self._array._writeable = new_writeable

    @property
    def fnc(self):
        return self.f_contiguous and not self.c_contiguous

    @property
    def forc(self):
        return self.f_contiguous or self.c_contiguous

    def __getitem__(self, name):
        if name == 'C_CONTIGUOUS':
            return self.c_contiguous
        elif name == 'F_CONTIGUOUS':
            return self.f_contiguous
        elif name == 'OWNDATA':
            return self.owndata
        elif name == 'WRITEABLE':
            return self.writeable
        else:
            raise KeyError('%s is not defined for cupy.ndarray.flags' % name)

    def __repr__(self):
        t = '  %s : %s'
        ret = []
        for name in 'C_CONTIGUOUS', 'F_CONTIGUOUS', 'OWNDATA', 'WRITEABLE':
            ret.append(t % (name, self[name]))
        return '\n'.join(ret)
