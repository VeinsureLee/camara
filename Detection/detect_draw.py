import cv2
from Detection.human_detect.human_detect.detect import detect_human  # 人体框检测函数，返回 boxes 和对应 frame
from Detection.human_face_recognition.human_face_recognition.face_detect import load_face_database, detect_faces  #
# 人脸识别函数（可作用于 frame）
from Detection.model.model.predict import pose_status  # 姿态识别函数（接受 frame 或 crop）
from Detection.model.model.config_ActionDetector import ActionDetectorConfig
from Detection.human_detect.human_detect.config_HumanDetector import HumanDetectorConfig
import torch

action_model_config = ActionDetectorConfig(
    labels=None,
    base_path=None,  # test database save path
    model_path='./model/model_pretrained/best_pose_cnn_model.pth',  # pretrained model path
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
    model_path='./human_detect/model/yolov8n.pt'
)


def human_action_detect(frame, human_enable, face_enable, action_enable,
                        known_face_encodings, known_face_names,
                        action_model_cfg=action_model_config,
                        human_model_cfg=human_model_config):
    """
    分析一帧图像：
    - 若启用人体检测（human_enable==True），则检测人体框；
    - 对每个人体框，计算中心位置；
    - 若启用人脸识别（face_enable==True），则识别框内的人脸，否则返回 'Nobody'；
    - 若启用姿态检测（action_enable==True），则识别动作状态，否则返回 'No action'；

    输出每个检测的人体信息，包含：
      - bbox：人像的边界框 (x1, y1, x2, y2)
      - center：边界框中心点 (cx, cy)
      - face_info：人脸信息（识别不到则为 "Nobody"）
      - action：动作识别结果（识别失败则为 "No action"）
    """
    detections = []

    if not human_enable:
        # 若人体检测没有启用，则返回空列表
        return detections

    # 检测人体框，得到结果
    results = detect_human(frame, human_model_cfg)
    if not results or not hasattr(results, 'boxes'):
        return detections

    for box in results.boxes:
        # 将检测框转换成整数坐标
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # 计算边界框中心坐标
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        # 从整张图片中裁剪人体区域
        crop = frame[y1:y2, x1:x2]

        # 人脸识别处理
        face_info = "Unknown face"
        if face_enable:
            try:
                faces = detect_faces(crop, known_face_encodings, known_face_names)
                if faces:
                    face_info = faces[0]['name']
            except Exception as e:
                # 如果识别出现异常，则保持默认值 "Nobody"
                face_info = "Unknown face"

        # 姿态识别处理
        action = "Unknown action"
        if action_enable:
            try:
                action = pose_status(crop, action_model_cfg)
            except Exception as e:
                action = "No action"

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "center": center,
            "face_info": face_info,
            "action": action
        })

    return detections


def draw_detection(frame,
                   human_enable, face_enable, action_enable,
                   known_face_encodings, known_face_names,
                   cfg1=action_model_config,
                   cfg2=human_model_config,
                   ):
    """
    输入：
        frame: 原始图像帧
        human_enable, face_enable, action_enable: 各识别功能开关
        known_face_encodings, known_face_names: 人脸识别所需数据库
    返回：
        - 原图 frame（未修改）
        - 画上检测结果的图像 frame_drawn
    """
    # 深拷贝一份图像用于绘制
    frame_drawn = frame.copy()

    detections = human_action_detect(
        frame,
        human_enable, face_enable, action_enable,
        known_face_encodings, known_face_names,
        cfg1, cfg2
    )

    for item in detections:
        x1, y1, x2, y2 = item["bbox"]
        center = item["center"]
        face_text = f"{item['face_info']}"
        action_text = f"{item['action']}"

        # 绘制边界框
        cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制两行文本，红色，小字体
        cv2.putText(frame_drawn, face_text, (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame_drawn, action_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 可选：绘制中心点
        # cv2.circle(frame_drawn, center, 5, (255, 0, 0), -1)

    return frame, frame_drawn


def run_camera(known_face_encodings, known_face_names,
               human_enable=True, face_enable=True, action_enable=True,
               cfg1=action_model_config, cfg2=human_model_config):
    """
    打开摄像头并实时处理帧，显示检测结果。
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_frame, processed_frame = draw_detection(
            frame, human_enable, face_enable, action_enable,
            known_face_encodings, known_face_names,
            cfg1, cfg2
        )

        cv2.imshow("Human Action Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 加载人脸数据库
    encodings, names = load_face_database('./human_face_recognition/facebase')

    # # 读取测试图片
    # img = cv2.imread('./pinshi.png')
    # cv2.imshow("Original Image", img)
    # cv2.waitKey(0)
    #
    # # 处理图像并获取绘制后的结果
    # _, img_pre = draw_detection(img, True, True, True, encodings, names)
    # cv2.imshow("Detection Result", img_pre)
    # cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()
    run_camera(encodings, names, True, False, True)
