import os
import socket
import sys
import winreg
import time
import io
import cv2
import numpy as np
import face_recognition
from PIL import Image, ImageFont, ImageDraw
from PySide6.QtCore import QTimer, QThread, QSettings, Qt
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QGroupBox,
    QHBoxLayout, QVBoxLayout, QLineEdit, QComboBox, QMessageBox, QFileDialog
)

CAPTURE_IMAGE_DATA = None
CAPTURE_IMAGE_DATA_TMP = None
SAVE_VIDEO_PATH = ""
FACE_DATABASE_PATH = "Detection/human_face_recognition/facebase"


def load_face_database(facebase_path):
    known_face_encodings = []
    known_face_names = []
    for person_name in os.listdir(facebase_path):
        person_path = os.path.join(facebase_path, person_name)
        if not os.path.isdir(person_path):
            continue
        person_encodings = []
        for image_name in os.listdir(person_path):
            if not image_name.lower().endswith((".jpg", ".png")):
                continue
            image_path = os.path.join(person_path, image_name)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                person_encodings.append(encodings[0])
        if person_encodings:
            known_face_encodings.append(np.mean(person_encodings, axis=0))
            known_face_names.append(person_name)
    return known_face_encodings, known_face_names


class CaptureThread(QThread):
    def __init__(self, ip, port):
        super().__init__()
        self.tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp_server_socket.bind((ip, port))
        self.tcp_server_socket.listen(128)
        self.run_flag = True
        self.record_flag = False

    def run(self):
        global CAPTURE_IMAGE_DATA, CAPTURE_IMAGE_DATA_TMP
        new_s, client_info = self.tcp_server_socket.accept()
        print("新的客户端链接：", client_info)

        while self.run_flag:
            length = self.receive_all(new_s, 16)
            if not length:
                break
            video_data = self.receive_all(new_s, int(length))
            try:
                bytes_stream = io.BytesIO(video_data)
                image = Image.open(bytes_stream)
                img = np.asarray(image)
                CAPTURE_IMAGE_DATA_NMP = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if self.record_flag:
                    self.mp4_file.write(CAPTURE_IMAGE_DATA_NMP)
                temp_image = QImage(img.flatten(), 480, 320, QImage.Format_RGB888)
                CAPTURE_IMAGE_DATA = QPixmap.fromImage(temp_image)
            except Exception as ret:
                print("error:", ret)

        new_s.close()
        self.tcp_server_socket.close()

    @staticmethod
    def receive_all(sock, count):
        buf = b''
        while count:
            recv_data_temp = sock.recv(count)
            if not recv_data_temp:
                return None
            buf += recv_data_temp
            count -= len(recv_data_temp)
        return buf

    def stop_run(self):
        self.run_flag = False
        self.record_flag = False
        try:
            self.mp4_file.release()
        except:
            pass

    def start_record(self):
        video_type = cv2.VideoWriter_fourcc(*'XVID')
        file_name = f"{time.time()}.avi"
        file_path = os.path.join(SAVE_VIDEO_PATH, file_name)
        self.mp4_file = cv2.VideoWriter(file_path, video_type, 5, (480, 320))
        self.record_flag = True

    def stop_record(self):
        self.record_flag = False
        self.mp4_file.release()


class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("远程摄像头监控+人脸识别")
        self.setWindowIcon(QIcon('./CarIdentityData/logo.png'))
        self.resize(777, 555)

        # 加载人脸数据库
        self.known_face_encodings, self.known_face_names = load_face_database(FACE_DATABASE_PATH)
        self.face_recognition_enabled = False
        self.font = ImageFont.truetype("./simhei.ttf", 30)

        # 界面组件初始化
        camera_label = QLabel("选择本电脑IP：")
        self.combox = QComboBox()
        hostname, _, ip_addr_list = socket.gethostbyname_ex(socket.gethostname())
        ip_addr_list.insert(0, "0.0.0.0")
        self.combox.addItems(ip_addr_list)
        port_label = QLabel("本地端口：")
        self.port_edit = QLineEdit("9000")

        # 第一组布局
        g_1 = QGroupBox("监听信息")
        g_1.setFixedHeight(60)
        g_1_layout = QHBoxLayout()
        g_1_layout.addWidget(camera_label)
        g_1_layout.addWidget(self.combox)
        g_1_layout.addWidget(port_label)
        g_1_layout.addWidget(self.port_edit)
        g_1.setLayout(g_1_layout)

        # 功能按钮
        self.camera_btn = QPushButton(QIcon("./CarIdentityData/shexiangtou.png"), "启动显示")
        self.camera_btn.clicked.connect(self.toggle_camera)
        self.record_btn = QPushButton(QIcon("./CarIdentityData/record.png"), "开始录制")
        self.record_btn.clicked.connect(self.toggle_record)
        self.path_btn = QPushButton(QIcon("./CarIdentityData/folder.png"), "设置路径")
        self.path_btn.clicked.connect(self.set_save_path)
        self.face_btn = QPushButton(QIcon("./CarIdentityData/face.png"), "启动识别")
        self.face_btn.clicked.connect(self.toggle_face_recognition)

        # 第二组布局
        g_2 = QGroupBox("功能操作")
        g_2.setFixedHeight(60)
        g_2_layout = QHBoxLayout()
        g_2_layout.addWidget(self.camera_btn)
        g_2_layout.addWidget(self.record_btn)
        g_2_layout.addWidget(self.path_btn)
        g_2_layout.addWidget(self.face_btn)
        g_2.setLayout(g_2_layout)

        # 视频显示区域
        self.video_label = QLabel("等待视频连接...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(True)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(g_1)
        main_layout.addWidget(g_2)
        main_layout.addWidget(self.video_label)
        self.setLayout(main_layout)

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def toggle_camera(self):
        if self.camera_btn.text() == "启动显示":
            ip = self.combox.currentText()
            try:
                port = int(self.port_edit.text())
            except:
                QMessageBox.critical(self, "错误", "端口号无效")
                return
            self.capture_thread = CaptureThread(ip, port)
            self.capture_thread.start()
            self.timer.start(30)
            self.camera_btn.setText("关闭显示")
        else:
            self.capture_thread.stop_run()
            self.timer.stop()
            self.video_label.clear()
            self.camera_btn.setText("启动显示")

    def toggle_record(self):
        if self.record_btn.text() == "开始录制":
            if not SAVE_VIDEO_PATH:
                QMessageBox.warning(self, "警告", "请先设置保存路径")
                return
            self.capture_thread.start_record()
            self.record_btn.setText("停止录制")
        else:
            self.capture_thread.stop_record()
            self.record_btn.setText("开始录制")

    def set_save_path(self):
        global SAVE_VIDEO_PATH
        path = QFileDialog.getExistingDirectory(self, "选择保存目录", os.path.expanduser("~"))
        if path:
            SAVE_VIDEO_PATH = path

    def toggle_face_recognition(self):
        self.face_recognition_enabled = not self.face_recognition_enabled
        self.face_btn.setText("停止识别" if self.face_recognition_enabled else "启动识别")

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        pil_image = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_image)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            name = "未知"
            if len(distances) > 0:
                best_match = np.argmin(distances)
                if distances[best_match] < 0.5:
                    name = self.known_face_names[best_match]
            draw.rectangle([(left, top), (right, bottom)], outline=(255,0,0), width=3)
            draw.text((left, top-30), name, font=self.font, fill=(255,0,0))

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def update_frame(self):
        global CAPTURE_IMAGE_DATA_TMP
        if CAPTURE_IMAGE_DATA_NMP is not None:
            if self.face_recognition_enabled:
                frame = self.process_frame(CAPTURE_IMAGE_DATA_NMP.copy())
            else:
                frame = CAPTURE_IMAGE_DATA_NMP

            h, w, ch = frame.shape
            bytes_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_line, QImage.Format_BGR888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

    @staticmethod
    def get_desktop():
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders')
        return winreg.QueryValueEx(key, "Desktop")[0]


def main():
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
