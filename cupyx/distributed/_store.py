import multiprocessing
import pickle
import threading
import socket
import time


class TCPStore:

    def __init__(self, world_size):
        self.storage = {}
        self._process = None
        self._run = multiprocessing.Value('b', True)
        self._world_size = {}

    def __del__(self):
        if self._process is not None:
            with self._run.get_lock():
                self._run.value = 0

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

    def run(self, rank, host='127.0.0.1', port=12345):
        if rank == 0:
            # Run the TCP store in a different process
            p = multiprocessing.Process(
                target=self._server_loop, args=(host, port))
            p.start()
            self._process = p


class TCPStoreProxy:

    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port

    def __getitem__(self, key):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # TODO retry connects
            s.connect((self.host, self.port))
            s.sendall(pickle.dumps(TCPStore.Get(key)))
            data = s.recv(1024)
        return pickle.loads(data)

    def __setitem__(self, key, value):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # TODO retry connects
            s.connect((self.host, self.port))
            s.sendall(pickle.dumps(TCPStore.Set(key, value)))


if __name__ == '__main__':
    store = TCPStore()
    store.run(0)
    print('creating proxy')
    proxy = TCPStoreProxy()
    time.sleep(1)
    proxy[123] = 24
    print(proxy[123])
    del store
