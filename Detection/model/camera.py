from model.predict import pose_status
import cv2
import mediapipe as mp
from Detection.model.model import mp_drawing, mp_pose, pose


def camera():
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

            # 显示检测结果
            text = pose_status(image_rgb)
            cv2.putText(frame, text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Pose Detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # 释放视频捕获对象
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    camera()
