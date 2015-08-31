class Flags(object):

    def __init__(self, c_contiguous, f_contiguous, owndata):
        self.c_contiguous = c_contiguous
        self.f_contiguous = f_contiguous
        self.owndata = owndata

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
