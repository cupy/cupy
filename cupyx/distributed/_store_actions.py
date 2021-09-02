import enum
import threading

from cupyx.distributed import _klv_utils


class Actions(enum.IntEnum):
    Set = 1
    Get = 2
    Barrier = 3


def execute_action(action, value, store):
    # receive the remaining amount of bytes that L field specifies
    try:
        if action == Actions.Set:
            action_obj = Set.from_klv(value)
        elif action == Actions.Get:
            action_obj = Get.from_klv(value)
        elif action == Actions.Barrier:
            action_obj = Barrier.from_klv(value)
        else:
            raise ValueError(f'unknown action {action}')
        return action_obj(store)
    except Exception as e:
        return _klv_utils.get_result_action_t(1, str(e).encode('ascii'))


class Set:
    class SetResult:
        def klv(self):
            v = bytearray(bytes(True))
            action = _klv_utils.get_result_action_t(0, v)
            return bytes(action)

        @staticmethod
        def from_klv(klv):
            return True

    def __init__(self, key, value):
        self.key = key
        self.value = value
        # Check, value can only be integer or bytes

    @staticmethod
    def from_klv(value):
        value = bytes(value)
        for i, b in enumerate(value):
            if b == 0:
                k = value[:i].decode('utf-8')
                value = value[i + 1:]
                break
        else:
            raise ValueError('No separation character for key found')
        v = _klv_utils.get_value_from_bytes(value)
        return Set(k, v)

    def klv(self):
        v = bytearray(self.key.encode('ascii'))
        v.append(0)  # marker to show where the value begins
        v += _klv_utils.create_value_bytes(self.value)
        action = _klv_utils.get_action_t(Actions.Set, v)
        return bytes(action)

    def __call__(self, store):
        store.storage[self.key] = self.value
        return Set.SetResult()

    def decode_result(self, data):
        return Set.SetResult.from_klv(data)


class Get:
    class GetResult:
        def __init__(self, value):
            self.value = value

        def klv(self):
            v = _klv_utils.create_value_bytes(self.value)
            action = _klv_utils.get_result_action_t(0, v)
            return bytes(action)

        @staticmethod
        def from_klv(value):
            value = bytearray(value)
            return _klv_utils.get_value_from_bytes(value)

    def __init__(self, key):
        self.key = key

    @staticmethod
    def from_klv(value):
        k = value.decode('utf-8')
        return Get(k)

    def klv(self):
        v = bytearray(self.key.encode('ascii'))
        action = _klv_utils.get_action_t(Actions.Get, v)
        return bytes(action)

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
            action = _klv_utils.get_result_action_t(0, v)
            return bytes(action)

        @staticmethod
        def from_klv(klv):
            # don't need to parse, barrier always sends True
            return True

    def klv(self):
        action = _klv_utils.get_action_t(Actions.Barrier, bytes(0))
        return bytes(action)

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
