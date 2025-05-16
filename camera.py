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
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QGroupBox, QHBoxLayout, QVBoxLayout, \
    QLineEdit, QComboBox, QMessageBox, QFileDialog
from Detection.car_recognition.car_recognition.PlateDetector import plate_detector, plate_detector_with_overlay
from Detection.model.model.config_ActionDetector import ActionDetectorConfig
from Detection.human_detect.human_detect.config_HumanDetector import HumanDetectorConfig
from Detection.human_face_recognition.human_face_recognition.face_detect import load_face_database
from Detection.detect_draw import draw_detection

CAPTURE_IMAGE_DATA = None
SAVE_VIDEO_PATH = ""
CAPTURE_IMAGE_DATA_TMP = None

HUMAN_ENABLE = False
FACE_ENABLE = False
ACTION_ENABLE = False
PLATE_ENABLE = False

plate_model_path = 'Detection/car_recognition/CarIdentityData/model/plate_recognize/plate_cnn_model.pth'
char_model_path = 'Detection/car_recognition/CarIdentityData/model/char_recognize/char_cnn_model.pth'

action_model_config = ActionDetectorConfig(
    labels=None,
    base_path=None,  # test database save path
    model_path='Detection/model/model_pretrained/best_pose_cnn_model.pth',  # pretrained model path
    save_path=None,
    num_epochs=None,
    root_total_input_folder=None,
    root_total_output_folder=None,
    root_test_input_folder=None,
    root_test_output_folder=None,
    train_log_path=None,
)

human_model_config = HumanDetectorConfig(
    img_path=None,
    model_path='Detection/human_detect/model/yolov8n.pt'
)

encodings, names = load_face_database('Detection/human_face_recognition/facebase')


class CaptureThread(QThread):
    def __init__(self, ip, port):
        super().__init__()
        self.tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        self.tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp_server_socket.bind((ip, port))
        self.tcp_server_socket.listen(128)
        self.run_flag = True
        self.record_flag = False

    def run(self):
        global CAPTURE_IMAGE_DATA
        global CAPTURE_IMAGE_DATA_TMP
        global PLATE_ENABLE
        global FACE_ENABLE
        global ACTION_ENABLE
        global HUMAN_ENABLE

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
                if PLATE_ENABLE == 1:
                    print("正在识别中")
                    img, _ = plate_detector_with_overlay(
                        img, plate_model_path, char_model_path
                    )
                    CAPTURE_IMAGE_DATA_TMP = img
                if HUMAN_ENABLE == 1:
                    print("正在识别中")
                    _, img, detections = draw_detection(img,
                                                        HUMAN_ENABLE, FACE_ENABLE, ACTION_ENABLE,
                                                        encodings, names,
                                                        action_model_config, human_model_config)
                    CAPTURE_IMAGE_DATA_TMP = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if self.record_flag:
                    self.mp4_file.write(CAPTURE_IMAGE_DATA_TMP)

                height, width, channel = img.shape
                bytes_per_line = 3 * width
                temp_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                CAPTURE_IMAGE_DATA = QPixmap.fromImage(temp_image)

                # temp_image = QImage(img.flatten(), 480, 320, QImage.Format_RGB888)
                # temp_pixmap = QPixmap.fromImage(temp_image)
                # CAPTURE_IMAGE_DATA = temp_pixmap

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

    def stop_record(self):
        self.mp4_file.release()
        self.record_flag = False

    def start_record(self):
        video_type = cv2.VideoWriter_fourcc(*'XVID')
        file_name = "{}.avi".format(time.time())
        file_path_name = os.path.join(SAVE_VIDEO_PATH, file_name)
        self.mp4_file = cv2.VideoWriter(file_path_name, video_type, 5, (480, 320))
        self.record_flag = True


class ShowCaptureVideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        self.video_label = QLabel("选择顶部的操作按钮...")
        self.video_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.video_label.setScaledContents(True)
        layout.addWidget(self.video_label)
        self.setLayout(layout)


class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("远程摄像头监控")
        self.setWindowIcon(QIcon('./CarIdentityData/logo.png'))
        self.resize(900, 580)

        # 监听设置区
        camera_label = QLabel("选择本电脑IP：")
        hostname, alias_list, ip_addr_list = socket.gethostbyname_ex(socket.gethostname())
        ip_addr_list.insert(0, "0.0.0.0")
        self.combox = QComboBox()
        self.combox.addItems(ip_addr_list)

        port_label = QLabel("本地端口：")
        self.port_edit = QLineEdit("9000")

        g_1 = QGroupBox("监听信息")
        g_1.setFixedHeight(60)
        g_1_h_layout = QHBoxLayout()
        g_1_h_layout.addWidget(camera_label)
        g_1_h_layout.addWidget(self.combox)
        g_1_h_layout.addWidget(port_label)
        g_1_h_layout.addWidget(self.port_edit)
        g_1.setLayout(g_1_h_layout)

        # 操作按钮区
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

        # 场景选择及功能按钮
        self.scene_button = QPushButton("选择场景")
        self.scene_button.clicked.connect(self.toggle_scene_options)

        self.scene_combo = QComboBox()
        self.scene_combo.addItems(["家居场景", "公共交通场景"])
        self.scene_combo.setVisible(False)
        self.scene_combo.currentTextChanged.connect(self.update_function_buttons)

        # 识别功能按钮
        self.person_locate_btn = QPushButton("开启人物位置识别")
        self.person_locate_btn.setVisible(False)
        self.person_locate_btn.clicked.connect(self.toggle_person_locate)

        self.face_rec_btn = QPushButton("开启人脸识别")
        self.face_rec_btn.setVisible(False)
        self.face_rec_btn.clicked.connect(self.toggle_face_rec)

        self.action_rec_btn = QPushButton("开启动作识别")
        self.action_rec_btn.setVisible(False)
        self.action_rec_btn.clicked.connect(self.toggle_action_rec)

        self.plate_rec_btn = QPushButton("开启车牌识别")
        self.plate_rec_btn.setVisible(False)
        self.plate_rec_btn.clicked.connect(self.toggle_plate_rec)

        for btn in [self.person_locate_btn, self.face_rec_btn, self.action_rec_btn, self.plate_rec_btn]:
            btn.setVisible(False)

        scene_layout = QVBoxLayout()
        scene_layout.addWidget(self.scene_button)
        scene_layout.addWidget(self.scene_combo)
        scene_layout.addWidget(self.person_locate_btn)
        scene_layout.addWidget(self.face_rec_btn)
        scene_layout.addWidget(self.action_rec_btn)
        scene_layout.addWidget(self.plate_rec_btn)

        self.scene_group = QGroupBox("场景选择与识别功能")
        self.scene_group.setLayout(scene_layout)
        self.scene_group.setFixedWidth(200)

        # 视频区域
        self.video_view = ShowCaptureVideoWidget()

        # 主界面布局
        top_layout = QHBoxLayout()
        top_layout.addWidget(g_1)
        top_layout.addWidget(g_2)
        top_layout.addStretch()

        left_layout = QVBoxLayout()
        left_layout.addLayout(top_layout)
        left_layout.addWidget(self.video_view)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.scene_group)

        self.setLayout(main_layout)

        # 定时刷新视频画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_video_image)
        self.load_time = 0
        self.load_time_all = 0

    def toggle_scene_options(self):
        self.scene_combo.setVisible(not self.scene_combo.isVisible())

    def update_function_buttons(self, scene):
        if scene == "家居场景":
            self.person_locate_btn.setVisible(True)
            self.face_rec_btn.setVisible(True)
            self.action_rec_btn.setVisible(True)
            self.plate_rec_btn.setVisible(False)
        elif scene == "公共交通场景":
            self.person_locate_btn.setVisible(False)
            self.face_rec_btn.setVisible(False)
            self.action_rec_btn.setVisible(False)
            self.plate_rec_btn.setVisible(True)

    def camera_open_close(self):
        if self.camera_open_close_btn.text() == "启动显示":
            ip = self.combox.currentText()
            try:
                port = int(self.port_edit.text())
            except:
                QMessageBox.about(self, '警告', '端口设置错误！！！')
                return

            self.thread = CaptureThread(ip, port)
            self.thread.daemon = True
            self.thread.start()
            self.timer.start(1)
            self.camera_open_close_btn.setText("关闭显示")
        else:
            self.camera_open_close_btn.setText("启动显示")
            self.timer.stop()
            self.video_view.video_label.clear()
            self.thread.stop_run()
            self.record_video_btn.setText("开始录制")

    def show_video_image(self):
        if CAPTURE_IMAGE_DATA:
            self.video_view.video_label.setPixmap(CAPTURE_IMAGE_DATA)
        else:
            if time.time() - self.load_time >= 1:
                self.load_time = time.time()
                self.load_time_all += 1
                self.video_view.video_label.setText(f"摄像头加载中...{self.load_time_all}")

    def get_desktop(self):
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders')
        return winreg.QueryValueEx(key, "Desktop")[0]

    def save_video_path_setting(self):
        global SAVE_VIDEO_PATH
        last_path = QSettings().value("LastFilePath") if SAVE_VIDEO_PATH else self.get_desktop()
        path_name = QFileDialog.getExistingDirectory(self, '请选择保存视频的路径', last_path)
        if path_name:
            SAVE_VIDEO_PATH = path_name

    def recorde_video(self):
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

    def toggle_person_locate(self):
        global HUMAN_ENABLE
        if self.person_locate_btn.text() == "开启人物位置识别":
            self.person_locate_btn.setText("关闭人物位置识别")
            # TODO: 启动人物位置识别逻辑
            print("启动人物位置识别逻辑")
            HUMAN_ENABLE = 1
        else:
            self.person_locate_btn.setText("开启人物位置识别")
            # TODO: 关闭人物位置识别逻辑
            print("关闭人物位置识别逻辑")
            HUMAN_ENABLE = 0

    def toggle_face_rec(self):
        global FACE_ENABLE
        if self.face_rec_btn.text() == "开启人脸识别":
            self.face_rec_btn.setText("关闭人脸识别")
            # TODO: 启动人脸识别逻辑
            print("启动人脸识别逻辑")
            FACE_ENABLE = 1
        else:
            self.face_rec_btn.setText("开启人脸识别")
            # TODO: 关闭人脸识别逻辑
            print("关闭人脸识别逻辑")
            FACE_ENABLE = 0

    def toggle_action_rec(self):
        global ACTION_ENABLE
        if self.action_rec_btn.text() == "开启动作识别":
            self.action_rec_btn.setText("关闭动作识别")
            # TODO: 启动动作识别逻辑
            print("启动动作识别逻辑")
            ACTION_ENABLE = 1
        else:
            self.action_rec_btn.setText("开启动作识别")
            # TODO: 关闭动作识别逻辑
            print("关闭动作识别逻辑")
            ACTION_ENABLE = 0

    def toggle_plate_rec(self):
        global PLATE_ENABLE
        if self.plate_rec_btn.text() == "开启车牌识别":
            self.plate_rec_btn.setText("关闭车牌识别")
            # TODO: 启动车牌识别逻辑
            print("启动车牌识别逻辑")
            PLATE_ENABLE = 1
        else:
            self.plate_rec_btn.setText("开启车牌识别")
            # TODO: 关闭车牌识别逻辑
            print("关闭车牌识别逻辑")
            PLATE_ENABLE = 0


def main():
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
