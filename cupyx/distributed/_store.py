import atexit
from ctypes import sizeof
import multiprocessing
import threading
import socket
import time

from cupyx.distributed import _klv_utils
from cupyx.distributed import _store_actions


_DEFAULT_HOST = '127.0.0.1'
_DEFAULT_PORT = 13333

_exit_mode = False


@atexit.register
def _exit():
    global _exit_mode
    _exit_mode = True


class ExceptionAwareProcess(multiprocessing.Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exception = None
        self._parent_p, self._child_p = multiprocessing.Pipe()

    def run(self):
        try:
            super().run()
            self._child_p.send(None)
        except Exception as e:
            self._child_p.send(e)

    def join(self):
        super().join()
        if self._parent_p.poll():
            exception = self._parent_p.recv()
            if exception is not None:
                raise exception


class TCPStore:
    # This is only used for initialization of nccl so we don't care
    # too much about peformance
    def __init__(self, world_size):
        self.storage = {}
        self._process = None
        self._world_size = world_size
        self._run = multiprocessing.Value('b', 1)
        # For implementing a barrier
        self._lock = threading.Lock()
        self._current_barrier = None

    def __del__(self):
        if not _exit_mode:
            self.stop()

    def _set_process(self, process):
        self._process = process

    def _process_request(self, c_socket):
        with c_socket:
            # Receive in KLV format
            action_bytes = c_socket.recv(sizeof(_klv_utils.action_t))
            if len(action_bytes) > 0:
                action_m = _klv_utils.action_t.from_buffer_copy(action_bytes)
                if action_m.length > 256:
                    raise ValueError('Invalid length for message')
                value = bytearray(action_m.value)[:action_m.length]
                r = _store_actions.execute_action(action_m.action, value, self)
                if r is not None:
                    c_socket.sendall(r.klv())

    def _server_loop(self, host, port):
        # This is for minimum info exchange during initialization
        # a single connection allows to implement locking mechanics easily
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen()
            s.settimeout(0.5)
            while self._run.value == 1:
                try:
                    c_socket, addr = s.accept()
                except socket.timeout:
                    continue

                t = threading.Thread(
                    target=self._process_request,
                    args=(c_socket,), daemon=True)
                t.start()

    def run(self, host=_DEFAULT_HOST, port=_DEFAULT_PORT):
        # Run the TCP store in a different process
        p = ExceptionAwareProcess(
            target=self._server_loop, args=(host, port))
        p.start()
        self._process = p

    def stop(self):
        if _exit_mode:
            return  # Prevent shutdown errors
        if self._process is not None:
            with self._run.get_lock():
                self._run.value = 0
            self._process.join()


class TCPStoreProxy:

    MAX_NUM_RETRIES = 50
    DELAY_FOR_RETRY = 0.5

    def __init__(self, host=_DEFAULT_HOST, port=_DEFAULT_PORT):
        self.host = host
        self.port = port

    def _send_recv(self, action):
        # Retry several times in case the rank 0 has not established the
        # main store yet
        for i in range(TCPStoreProxy.MAX_NUM_RETRIES):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    # TODO retry connects
                    s.connect((self.host, self.port))
                    s.sendall(action.klv())
                    result_bytes = s.recv(sizeof(
                        _klv_utils.result_action_t))
                    if len(result_bytes) > 0:
                        result = _klv_utils.result_action_t.from_buffer_copy(
                            result_bytes)
                        value = bytearray(result.value)[:result.length]
                        if result.status == 0:
                            return action.decode_result(value)
                        else:
                            raise RuntimeError(value.decode('utf-8'))
            except ConnectionRefusedError:
                time.sleep(TCPStoreProxy.DELAY_FOR_RETRY)
        raise RuntimeError('TCPStore is not available')

    def __getitem__(self, key):
        return self._send_recv(_store_actions.Get(key))

    def __setitem__(self, key, value):
        self._send_recv(_store_actions.Set(key, value))

    def barrier(self):
        # Barrier has special semantics
        self._send_recv(_store_actions.Barrier())
