import os
import socket
import sys
import winreg
import time
import io

import cv2
from PIL import Image
import numpy as np
from PySide6.QtCore import QTimer, QThread, QSettings
from PySide6.QtGui import QIcon, Qt, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QGroupBox, QHBoxLayout, QStyle, QVBoxLayout, QStackedLayout, QLineEdit, QComboBox, QMessageBox, QFileDialog

CAPTURE_IMAGE_DATA = None
SAVE_VIDEO_PATH = ""


class CaptureThread(QThread):
    def __init__(self, ip, port):
        super().__init__()
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
        self.udp_socket.bind((ip, port))
        # 设置运行的标志
        self.run_flag = True
        self.record_flag = False

    def run(self):
        global CAPTURE_IMAGE_DATA

        while self.run_flag:
            data, ip = self.udp_socket.recvfrom(100000)
            bytes_stream = io.BytesIO(data)
            image = Image.open(bytes_stream)
            img = np.asarray(image)

            if self.record_flag:  # 开始录制视频
                # 转换数据格式，便于存储视频文件
                img_2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # ESP32采集的是RGB格式，要转换为BGR（opencv的格式）
                self.mp4_file.write(img_2)

            # PySide显示不需要转换，所以直接用img
            temp_image = QImage(img.flatten(), 480, 320, QImage.Format_RGB888)
            temp_pixmap = QPixmap.fromImage(temp_image)
            CAPTURE_IMAGE_DATA = temp_pixmap  # 暂时存储udp接收到的1帧视频画面

        # 结束后 关闭套接字
        self.udp_socket.close()

    def stop_run(self):
        self.run_flag = False
        self.record_flag = False
        try:
            self.mp4_file.release()
        except Exception as ret:
            pass

    def stop_record(self):
        self.mp4_file.release()
        self.record_flag = False

    def start_record(self):
        # 设置视频的编码解码方式avi
        video_type = cv2.VideoWriter_fourcc(*'XVID')  # 视频存储的格式
        # 保存的位置，以及编码解码方式，帧率，视频帧大小
        file_name = "{}.avi".format(time.time())
        file_path_name = os.path.join(SAVE_VIDEO_PATH, file_name)
        self.mp4_file = cv2.VideoWriter(file_path_name, video_type, 5, (480, 320))
        self.record_flag = True


class ShowCaptureVideoWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()

        # 用来显示画面的QLabel
        self.video_label = QLabel("选择顶部的操作按钮...")
        self.video_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.video_label.setScaledContents(True)
        layout.addWidget(self.video_label)

        self.setLayout(layout)


class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("远程摄像头监控 v2022.10.13.001")
        self.setWindowIcon(QIcon('./CarIdentityData/logo.png'))
        self.resize(777, 555)

        # 选择本电脑IP
        camera_label = QLabel("选择本电脑IP：")

        # ip列表
        # 获取本地电脑的ip地址列表
        hostname, alias_list, ip_addr_list = socket.gethostbyname_ex(socket.gethostname())
        # print(hostname)  # DESKTOP
        # print(alias_list)  # []
        # print(ip_addr_list)  # ['192.168.47.1', '192.168.208.1', '192.168.31.53']
        ip_addr_list.insert(0, "0.0.0.0")
        self.combox = QComboBox()
        self.combox.addItems(ip_addr_list)
        self.ip_addr_list = ip_addr_list

        # 本地端口
        port_label = QLabel("本地端口：")
        self.port_edit = QLineEdit("9090")

        g_1 = QGroupBox("监听信息")
        g_1.setFixedHeight(60)
        g_1_h_layout = QHBoxLayout()
        g_1_h_layout.addWidget(camera_label)
        g_1_h_layout.addWidget(self.combox)
        g_1_h_layout.addWidget(port_label)
        g_1_h_layout.addWidget(self.port_edit)
        g_1.setLayout(g_1_h_layout)

        # 启动显示
        self.camera_open_close_btn = QPushButton(QIcon("./CarIdentityData/shexiangtou.png"), "启动显示")
        self.camera_open_close_btn.clicked.connect(self.camera_open_close)

        self.record_video_btn = QPushButton(QIcon("./CarIdentityData/record.png"), "开始录制")
        self.record_video_btn.clicked.connect(self.recorde_video)

        save_video_path_setting_btn = QPushButton(QIcon("./CarIdentityData/folder.png"), "设置保存路径")
        save_video_path_setting_btn.clicked.connect(self.save_video_path_setting)

        g_2 = QGroupBox("功能操作")
        g_2.setFixedHeight(60)
        g_2_h_layout = QHBoxLayout()
        g_2_h_layout.addWidget(self.camera_open_close_btn)
        g_2_h_layout.addWidget(self.record_video_btn)
        g_2_h_layout.addWidget(save_video_path_setting_btn)
        g_2.setLayout(g_2_h_layout)

        # --------- 整体布局 ---------
        h_layout = QHBoxLayout()
        h_layout.addWidget(g_1)
        h_layout.addWidget(g_2)
        h_layout.addStretch(1)

        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)

        # 创建底部的显示区域
        self.stacked_layout_capture_view = ShowCaptureVideoWidget()
        v_layout.addWidget(self.stacked_layout_capture_view)

        self.setLayout(v_layout)

        # 定时刷新视频画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_video_image)
        self.load_time = 0
        self.load_time_all = 0

    def camera_open_close(self):
        """启动创建socket线程，用来接收显示数据"""
        if self.camera_open_close_btn.text() == "启动显示":
            ip = self.combox.currentText()
            try:
                port = int(self.port_edit.text())
            except Exception as ret:
                QMessageBox.about(self, '警告', '端口设置错误！！！')
                return

            self.thread = CaptureThread(ip, port)
            self.thread.daemon = True
            self.thread.start()
            self.timer.start(100)  # 设置计时间隔并启动
            self.camera_open_close_btn.setText("关闭显示")
        else:
            self.camera_open_close_btn.setText("启动显示")
            self.timer.stop()
            self.stacked_layout_capture_view.video_label.clear()
            self.thread.stop_run()
            self.record_video_btn.setText("开始录制")

    def show_video_image(self):
        if CAPTURE_IMAGE_DATA:
            self.stacked_layout_capture_view.video_label.setPixmap(CAPTURE_IMAGE_DATA)
        else:
            if time.time() - self.load_time >= 1:
                self.load_time = time.time()
                self.load_time_all += 1
                self.stacked_layout_capture_view.video_label.setText("摄像头加载中...{}".format(self.load_time_all))

    @staticmethod
    def get_desktop():
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders')
        return winreg.QueryValueEx(key, "Desktop")[0]

    def save_video_path_setting(self):
        """视频保存路径"""
        global SAVE_VIDEO_PATH
        if SAVE_VIDEO_PATH:
            last_path = QSettings().value("LastFilePath")
        else:
            last_path = self.get_desktop()

        path_name = QFileDialog.getExistingDirectory(self, '请选择保存视频的路径', last_path)
        if not path_name:
            return

        SAVE_VIDEO_PATH = path_name

    def recorde_video(self):
        """录制视频"""
        if self.camera_open_close_btn.text() == "启动显示":
            QMessageBox.about(self, '警告', '请先启动显示，然后再开始录制！！！')
            return

        if not SAVE_VIDEO_PATH:
            QMessageBox.about(self, '警告', '请先配置视频保存路径！！！')
            return

        if self.record_video_btn.text() == "开始录制":
            self.record_video_btn.setText("停止录制")
            self.thread.start_record()
        else:
            self.record_video_btn.setText("开始录制")
            self.thread.stop_record()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_window = VideoWindow()
    video_window.show()
    app.exec()
