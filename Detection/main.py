from Detection.human_face_recognition.human_face_recognition.face_detect import load_face_database
from Detection.detect_draw import run_camera


if __name__ == "__main__":
    # 加载人脸数据库
    encodings, names = load_face_database('./human_face_recognition/facebase')
    # 默认全部开启人体检测、人脸识别和姿态识别
    run_camera(encodings, names, human_enable=True, face_enable=True, action_enable=True)
