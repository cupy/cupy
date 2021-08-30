import multiprocessing
import pickle
import threading
import socket
import time


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

    class Set:
        def __init__(self, key, value):
            self.key = key
            self.value = value

        def __call__(self, store):
            store.storage[self.key] = self.value

    class Get:
        def __init__(self, key):
            self.key = key

        def __call__(self, store):
            return store.storage[self.key]

    class Exit:
        def __call__(self, store):
            store._run = False

    class Barrier:
        def __call__(self, store):
            with store._lock:
                if store._current_barrier is None:
                    store._current_barrier = _BarrierImpl(store._world_size)
            store._current_barrier()
            # Once the barrier has been completed, just clean it
            with store._lock:
                store._current_barrier = None
            return True

    def _set_process(self, process):
        self._process = process

    def _process_request(self, c_socket):
        with c_socket:
            data = c_socket.recv(1024)
            r = pickle.loads(data)(self)
            if r is not None:
                c_socket.sendall(pickle.dumps(r))

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
                    s.sendall(pickle.dumps(action))
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
                    s.sendall(pickle.dumps(action))
                    data = s.recv(1024)
                    return pickle.loads(data)
            except ConnectionRefusedError:
                time.sleep(TCPStoreProxy.DELAY_FOR_RETRY)
        raise RuntimeError('TCPStore is not available')

    def __getitem__(self, key):
        return self._send_recv(TCPStore.Get(key))

    def __setitem__(self, key, value):
        self._send(TCPStore.Set(key, value))

    def barrier(self):
        # Barrier has special semantics
        self._send_recv(TCPStore.Barrier())
