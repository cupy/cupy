import threading


class Set:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        # Check, value can only be integer or bytes

    @staticmethod
    def from_klv(klv):
        for i, b in enumerate(klv):
            if b == 0:
                k = klv[:i].decode('utf-8')
                klv = klv[i + 1:]
                break
        else:
            raise ValueError('No separation character for key found')
        if klv[0:1] == 'i'.encode('ascii'):
            assert len(klv[1:]) == 8
            v = int.from_bytes(klv[1:], 'big')
        if klv[0:1] == 'b'.encode('ascii'):
            v = bytes(klv[1:])
        return Set(k, v)

    def klv(self):
        k = bytearray('set'.encode('ascii'))
        v = bytearray(self.key.encode('ascii'))
        v.append(0)  # marker to show where the value begins
        if type(self.value) is bytes:
            v = v + bytearray('b'.encode('ascii'))
            v = v + bytearray(self.value)
        elif type(self.value) is int:
            v = v + bytearray('i'.encode('ascii'))
            v = v + bytearray(self.value.to_bytes(8, byteorder='big'))
        else:
            raise ValueError(f'invalid type for self.value {self.value}')
        le = len(v).to_bytes(8, byteorder='big')
        return bytes(k + bytearray(le) + v)

    def __call__(self, store):
        store.storage[self.key] = self.value


class Get:
    class GetResult:
        def __init__(self, value):
            self.value = value

        def klv(self):
            if type(self.value) is bytes:
                k = bytearray('b'.encode('ascii'))
                v = bytearray(self.value)
            elif type(self.value) is int:
                k = bytearray('i'.encode('ascii'))
                v = bytearray(self.value.to_bytes(8, byteorder='big'))
            else:
                raise ValueError('invalid type for self.value')
            le = len(v).to_bytes(8, byteorder='big')
            return bytes(k + bytearray(le) + v)

        @staticmethod
        def from_klv(klv):
            klv = bytearray(klv)
            assert len(klv[9:]) == int.from_bytes(klv[1:9], 'big')
            if klv[0:1] == 'i'.encode('ascii'):
                v = int.from_bytes(klv[9:], 'big')
            if klv[0:1] == 'b'.encode('ascii'):
                v = bytes(klv[9:])
            return v

    def __init__(self, key):
        self.key = key

    @staticmethod
    def from_klv(klv):
        for i, b in enumerate(klv):
            if b == 0:
                k = klv[:i].decode('utf-8')
                break
        else:
            raise ValueError('No separation character for key found')
        return Get(k)

    def klv(self):
        k = bytearray('get'.encode('ascii'))
        v = bytearray(self.key.encode('ascii'))
        v.append(0)  # marker to show where the value begins
        le = len(v).to_bytes(8, byteorder='big')
        return bytes(k + bytearray(le) + v)

    def __call__(self, store):
        # TODO - return in KLV too
        # wrap into GetResult class with klv, from_klv_methods
        return Get.GetResult(store.storage[self.key])

    def decode_result(self, data):
        return Get.GetResult.from_klv(data)


class _BarrierImpl:
    def __init__(self, world_size):
        self._world_size = world_size
        self._cvar = threading.Condition()

    def __call__(self):
        # Superlame implementation, should be improved
        with self._cvar:
            self._world_size -= 1
            if self._world_size == 0:
                self._cvar.notifyAll()
            elif self._world_size > 0:
                self._cvar.wait()


class Barrier:
    class BarrierResult:
        def klv(self):
            v = bytearray(bytes(True))
            le = len(v).to_bytes(8, byteorder='big')
            return bytes(bytearray(le) + v)

        @staticmethod
        def from_klv(klv):
            # don't need to parse, barrier always sends True
            return True

    def klv(self):
        k = bytearray('bar'.encode('ascii'))
        le = len([]).to_bytes(8, byteorder='big')
        return bytes(k + bytearray(le))

    @staticmethod
    def from_klv(klv):
        return Barrier()

    def __call__(self, store):
        with store._lock:
            if store._current_barrier is None:
                store._current_barrier = _BarrierImpl(store._world_size)
        store._current_barrier()
        # Once the barrier has been completed, just clean it
        with store._lock:
            store._current_barrier = None
        return Barrier.BarrierResult()

    def decode_result(self, data):
        return Barrier.BarrierResult.from_klv(data)
