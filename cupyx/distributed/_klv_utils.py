def split_klv(klv, key_len=3, l_len=8):
    k = klv[0:key_len].decode('utf-8')
    le = int.from_bytes(klv[key_len:key_len + l_len], 'big')
    return k, le, klv[key_len + l_len:]


def create_value_bytes(value):
    if type(value) is bytes:
        v = bytearray('b'.encode('ascii'))
        v = v + bytearray(value)
    elif type(value) is int:
        v = bytearray('i'.encode('ascii'))
        v = v + bytearray(value.to_bytes(8, byteorder='big'))
    else:
        raise ValueError(f'invalid type for self.value {value}')
    return v


def get_value_from_bytes(v):
    if v[0:1] == 'i'.encode('ascii'):
        assert len(v[1:]) == 8
        v = int.from_bytes(v[1:], 'big')
    if v[0:1] == 'b'.encode('ascii'):
        v = bytes(v[1:])
    return v
