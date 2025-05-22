from typing import Optional
import socket, threading, io, time
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
        print(f"[Receiver] Listening on {self._ip}:{self._port}")

        while True:
            conn, addr = srv.accept()
            print(f"[Receiver] New connection from {addr}")
            conn.settimeout(self._timeout)

            try:
                while True:
                    length_bytes = self._recv_all(conn, 16)
                    if length_bytes is None:
                        print("[Receiver] Failed to receive length.")
                        break
                    try:
                        length = int(length_bytes.decode().strip())
                    except ValueError:
                        print("[Receiver] Invalid length header:", length_bytes)
                        break
                    data = self._recv_all(conn, length)
                    if data:
                        self.latest_frame = data
            except Exception as e:
                print(f"[Receiver] Connection error: {e}")
            finally:
                conn.close()
                print("[Receiver] Connection closed, waiting for new connection...")

    def _recv_all(self, sock, count):
        buf = b""
        while count:
            try:
                chunk = sock.recv(count)
                if not chunk:
                    return None
                buf += chunk
                count -= len(chunk)
            except socket.timeout:
                return None
        return buf


receiver = FrameReceiver()  # 全局实例
receiver.start()
