import os
import cv2
import numpy as np
from . import mp_pose, mp_drawing, test_photo_root_path, pose


def calculate_angle(a, b, c):
    """计算三个点之间的关节角度"""
    a = np.array(a)  # 起点 (肩)
    b = np.array(b)  # 中点 (肘)
    c = np.array(c)  # 终点 (腕)

    # 计算向量
    ba = a - b
    bc = c - b

    # 计算角度
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def is_climbing_linear(landmarks, image_shape):
    """攀爬动作判断函数"""
    # 关键点索引
    LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
    RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
    LEFT_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW
    RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW
    LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST
    RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST
    LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP
    RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP
    LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE
    RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE
    NOSE = mp_pose.PoseLandmark.NOSE

    # 获取归一化坐标并转换为像素坐标
    def get_coord(landmark):
        return [
            int(landmark.x * image_shape[1]),
            int(landmark.y * image_shape[0])
        ]

    try:
        # 头部位置（使用鼻子作为参考）
        head = get_coord(landmarks[NOSE])

        # 计算左臂角度
        left_shoulder = get_coord(landmarks[LEFT_SHOULDER])
        left_elbow = get_coord(landmarks[LEFT_ELBOW])
        left_wrist = get_coord(landmarks[LEFT_WRIST])
        angle_left_arm = calculate_angle(left_shoulder, left_elbow, left_wrist)

        # 计算右臂角度
        right_shoulder = get_coord(landmarks[RIGHT_SHOULDER])
        right_elbow = get_coord(landmarks[RIGHT_ELBOW])
        right_wrist = get_coord(landmarks[RIGHT_WRIST])
        angle_right_arm = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # 手腕高度判断（y坐标向下增大）
        wrist_above_head = (
                left_wrist[1] < head[1] or  # 左手腕高于头部
                right_wrist[1] < head[1]  # 右手腕高于头部
        )

        # 腿部弯曲检测（右腿）
        right_hip = get_coord(landmarks[RIGHT_HIP])
        right_knee = get_coord(landmarks[RIGHT_KNEE])
        right_ankle = get_coord(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])
        angle_right_leg = calculate_angle(right_hip, right_knee, right_ankle)

        # 腿部弯曲检测（左腿）
        left_hip = get_coord(landmarks[LEFT_HIP])
        left_knee = get_coord(landmarks[LEFT_KNEE])
        left_ankle = get_coord(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
        angle_left_leg = calculate_angle(left_hip, left_knee, left_ankle)

        # 攀爬判断条件（可根据实际需求调整阈值）
        condition_arm = (angle_left_arm > 160 or angle_right_arm > 160)  # 手臂接近伸直
        condition_leg = (angle_right_leg < 110 or angle_left_leg < 110)  # 腿部明显弯曲
        condition_height = wrist_above_head  # 手腕高于头部

        # 至少满足两个条件判定为攀爬动作
        return sum([condition_arm, condition_leg, condition_height]) >= 2

    except Exception as e:
        print(f"关键点缺失: {str(e)}")
        return False


def img_read(path=test_photo_root_path):
    for photo in os.listdir(path):
        img = cv2.imread(os.path.join(path, photo))
        if photo is None:
            print(f"无法读取图片: {os.path.join(path, photo)}")
            return
        photo_skl(img)


def photo_skl(image):
    results = pose.process(image)

    if results.pose_landmarks:
        # 绘制骨架
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 执行攀爬检测
        climbing_status = is_climbing_linear(
            results.pose_landmarks.landmark,
            image.shape
        )

        # 显示检测结果
        text = "Climbing Detected!" if climbing_status else "No Climbing Action"
        cv2.putText(image, text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Pose Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pose.close()
    return image


def camera_detect_skl():
    # 从摄像头读取视频流
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将图像转换为 RGB 格式
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # 绘制识别结果
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # 执行攀爬检测
            climbing_status = is_climbing_linear(
                results.pose_landmarks.landmark,
                frame.shape
            )
            # 显示检测结果
            text = "Climbing Detected!" if climbing_status else "No Climbing Action"
            cv2.putText(frame, text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Pose Detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # 释放视频捕获对象
    cap.release()
    cv2.destroyAllWindows()