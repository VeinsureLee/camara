# video_stream/tcp_receiver.py

import socket
import io
import threading
from PIL import Image
import numpy as np
import cv2

shared_frame = None


class TCPVideoReceiver(threading.Thread):
    def __init__(self, ip='0.0.0.0', port=9000):
        super().__init__()
        self.ip = ip
        self.port = port
        self.running = False

    def receive_all(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def run(self):
        global shared_frame
        self.running = True
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.ip, self.port))
        server.listen(1)
        print("Waiting for connection...")
        conn, addr = server.accept()
        print("Client connected:", addr)

        while self.running:
            try:
                length = self.receive_all(conn, 16)
                if not length:
                    break
                data = self.receive_all(conn, int(length))
                image = Image.open(io.BytesIO(data))
                frame = np.array(image)
                shared_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print("Error:", e)
                break

        conn.close()
        server.close()

    def stop(self):
        self.running = False
