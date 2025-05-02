import cv2
from Detection.human_detect.human_detect.detect import detect_human  # 人体框检测函数，返回 boxes 和对应 frame
from Detection.human_face_recognition.human_face_recognition.face_detect import load_face_database, detect_faces  #
# 人脸识别函数（可作用于 frame）
from Detection.model.model.predict import pose_status  # 姿态识别函数（接受 frame 或 crop）
from Detection.model.model.config_ActionDetector import ActionDetectorConfig
from Detection.human_detect.human_detect.config_HumanDetector import HumanDetectorConfig
from Detection.model.model import PoseCNN
import torch
action_model_config = ActionDetectorConfig(

    base_path=None,  # test database save path
    labels=None,
    model_path='./model/model_pretrained/best_pose_cnn_model.pth',
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
    model_path='./human_detect/model/yolov8n.pt',
)


def human_action_detect(frame, human_enable, face_enable, action_enable, known_face_encodings, known_face_names):
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
    results = detect_human(frame, human_model_config)
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
                action = pose_status(crop, action_model_config)
            except Exception as e:
                action = "No action"

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "center": center,
            "face_info": face_info,
            "action": action
        })

    return detections


def test(frame, known_face_encodings, known_face_names, human_enable=True, face_enable=True, action_enable=True):
    detections = human_action_detect(frame, human_enable, face_enable, action_enable, known_face_encodings,
                                     known_face_names)
    while True:
        for item in detections:
            x1, y1, x2, y2 = item["bbox"]
            center = item["center"]
            label = f"{item['face_info']} | {item['action']}"
            # 绘制人体边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 在边界框上方绘制识别信息
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # 绘制中心点
            # cv2.circle(frame, center, 5, (255, 0, 0), -1)

        cv2.imshow("Human Action Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return


def run_camera(known_face_encodings, known_face_names, human_enable=True, face_enable=True, action_enable=True):
    """
    使用摄像头实时采集视频帧，并调用 humanactiondetect 对每一帧进行处理，
    在图像上绘制人体边界框、中心位置、识别到的人脸信息以及动作信息。
    按下 'q' 键退出。
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = human_action_detect(frame, human_enable, face_enable, action_enable, known_face_encodings,
                                       known_face_names)
        for item in detections:
            x1, y1, x2, y2 = item["bbox"]
            center = item["center"]
            label = f"{item['face_info']} | {item['action']}"
            # 绘制人体边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 在边界框上方绘制识别信息
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # 绘制中心点
            # cv2.circle(frame, center, 5, (255, 0, 0), -1)

        cv2.imshow("Human Action Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 加载人脸数据库
    encodings, names = load_face_database('./human_face_recognition/facebase')
    # 默认全部开启人体检测、人脸识别和姿态识别
    run_camera(encodings, names, human_enable=True, face_enable=True, action_enable=True)
