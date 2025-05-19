import socket, threading, io, time
from typing import Optional

from PIL import Image


class FrameReceiver(threading.Thread):
    """
    后台线程：TCP → latest_frame (bytes)
    """
    def __init__(self, ip="0.0.0.0", port=9000, timeout=5):
        super().__init__(daemon=True)
        self._ip, self._port, self._timeout = ip, port, timeout
        self.latest_frame: Optional[bytes] = None

    def run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self._ip, self._port))
        srv.listen(1)
        conn, _ = srv.accept()
        conn.settimeout(self._timeout)
        while True:
            try:
                length = int(conn.recv(16))
                data = self._recv_all(conn, length)
                self.latest_frame = data
            except Exception:
                break  # 客户端断开 → 线程结束，可重启

    def _recv_all(self, sock, count):
        buf = b""
        while count:
            chunk = sock.recv(count)
            if not chunk:
                raise ConnectionError
            buf += chunk
            count -= len(chunk)
        return buf


receiver = FrameReceiver()  # 全局实例
receiver.start()
