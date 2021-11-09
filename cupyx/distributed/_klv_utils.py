from ctypes import Structure, c_int, c_byte


_VALUE_BUFFER_SIZE = 256


class action_t(Structure):
    _fields_ = [
        ('action', c_int),
        ('length', c_int),
        ('value', c_byte * _VALUE_BUFFER_SIZE)]


class result_action_t(Structure):
    _fields_ = [
        # If error value contains a message
        ('status', c_int),  # 0 OK, 1 Error
        ('length', c_int),
        ('value', c_byte * _VALUE_BUFFER_SIZE)]


def get_action_t(action, value):
    l_v = len(value)
    value = bytearray(value) + bytearray(
        b'\x00' * (_VALUE_BUFFER_SIZE - len(value)))
    value = (c_byte * _VALUE_BUFFER_SIZE).from_buffer(value)
    assert (_VALUE_BUFFER_SIZE - len(value)) >= 0
    return action_t(action, l_v, value)


def get_result_action_t(status, value):
    l_v = len(value)
    value = bytearray(value) + bytearray(
        b'\x00' * (_VALUE_BUFFER_SIZE - len(value)))
    value = (c_byte * _VALUE_BUFFER_SIZE).from_buffer(value)
    assert (_VALUE_BUFFER_SIZE - len(value)) >= 0
    return result_action_t(status, l_v, value)


def create_value_bytes(value):
    if type(value) is bytes:
        v = bytearray(b'b')
        v = v + bytearray(value)
    elif type(value) is int:
        v = bytearray(b'i')
        v = v + bytearray(value.to_bytes(8, byteorder='big'))
    else:
        raise ValueError(f'invalid type for self.value {value}')
    return v


def get_value_from_bytes(v):
    if v[0:1] == b'i':
        assert len(v[1:]) == 8
        v = int.from_bytes(v[1:], 'big')
    elif v[0:1] == b'b':
        v = bytes(v[1:])
    return v
