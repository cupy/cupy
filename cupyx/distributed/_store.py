import multiprocessing
import threading
import socket
import time

from cupyx.distributed import _klv_utils
from cupyx.distributed import _store_actions


class TCPStore:
    # This is only used for initialization of nccl so we don't care
    # too much about peformance
    def __init__(self, world_size):
        self.storage = {}
        self._process = None
        self._run = multiprocessing.Value('b', True)
        self._world_size = world_size
        # For implementing a barrier
        self._lock = threading.Lock()
        self._current_barrier = None

    def __del__(self):
        if self._process is not None:
            with self._run.get_lock():
                self._run.value = 0
            self._process.join()

    def _set_process(self, process):
        self._process = process

    def _process_request(self, c_socket):
        with c_socket:
            # Receive in KLV format
            klv = c_socket.recv(1024)
            if len(klv) > 0:
                klv = bytearray(klv)
                # receive the remaining amount of bytes that L field specifies
                k, le, v = _klv_utils.split_klv(klv)
                if le + 3 + 8 > 1024:
                    remaining = (le + 3 + 8) - 1024
                    v += bytearray(c_socket.recv(remaining))
                if le != len(v):
                    raise ValueError('Invalid payload length')
                if k == "set":
                    action = _store_actions.Set.from_klv(v)
                elif k == "get":
                    action = _store_actions.Get.from_klv(v)
                elif k == "bar":
                    assert le == 0
                    action = _store_actions.Barrier()
                else:
                    raise ValueError(f'unknown action {k}')
                r = action(self)
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
                    target=self._process_request, args=(c_socket,))
                t.setDaemon(True)
                t.start()

    def run(self, host='127.0.0.1', port=12345):
        # Run the TCP store in a different process
        p = multiprocessing.Process(
            target=self._server_loop, args=(host, port))
        p.start()
        self._process = p


class TCPStoreProxy:

    MAX_NUM_RETRIES = 50
    DELAY_FOR_RETRY = 0.5

    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port

    def _send(self, action):
        # Retry several times in case the rank 0 has not established the
        # main store yet
        for i in range(TCPStoreProxy.MAX_NUM_RETRIES):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    # TODO retry connects
                    s.connect((self.host, self.port))
                    s.sendall(action.klv())
                    return
            except ConnectionRefusedError:
                time.sleep(TCPStoreProxy.DELAY_FOR_RETRY)
        raise RuntimeError('TCPStore is not available')

    def _send_recv(self, action):
        # Retry several times in case the rank 0 has not established the
        # main store yet
        for i in range(TCPStoreProxy.MAX_NUM_RETRIES):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    # TODO retry connects
                    s.connect((self.host, self.port))
                    s.sendall(action.klv())
                    data = s.recv(1024)
                    return action.decode_result(data)
            except ConnectionRefusedError:
                time.sleep(TCPStoreProxy.DELAY_FOR_RETRY)
        raise RuntimeError('TCPStore is not available')

    def __getitem__(self, key):
        return self._send_recv(_store_actions.Get(key))

    def __setitem__(self, key, value):
        self._send(_store_actions.Set(key, value))

    def barrier(self):
        # Barrier has special semantics
        self._send_recv(_store_actions.Barrier())
