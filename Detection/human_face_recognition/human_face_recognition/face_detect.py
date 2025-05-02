import os
import cv2
import numpy as np
import face_recognition
from . import facebase_path


def load_face_database(facebase_path=facebase_path):
    """
        加载人脸数据库，对每个文件夹下的图片获取人脸编码，
        并计算各自的平均编码，返回已知人脸编码和对应人名
    """
    print(f'Loading face database from {facebase_path}...')
    known_face_encodings = []
    known_face_names = []
    for person_name in os.listdir(facebase_path):
        person_path = os.path.join(facebase_path, person_name)
        if not os.path.isdir(person_path):
            continue

        person_encodings = []
        for image_name in os.listdir(person_path):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
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


def detect_faces(frame, known_face_encodings, known_face_names):
    # 转换颜色空间从BGR到RGB（face_recognition 需要）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 检测人脸位置和编码
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_info = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 计算与已知人脸的相似度距离
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        name = "未知"

        if face_distances.size > 0:
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < 0.5:
                name = known_face_names[best_match_index]

        face_info.append({
            "location": (top, right, bottom, left),
            "name": name
        })

    return face_info


def display_faces(frame, face_info, ft_renderer=None):
    for info in face_info:
        top, right, bottom, left = info["location"]
        name = info["name"]

        # 画框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)

        # 显示中文名字
        if ft_renderer:
            ft_renderer.putText(frame, name, (left + 2, bottom - 5), 24, (0, 0, 0), thickness=-1, line_type=cv2.LINE_AA)
        else:
            # 若无法使用 freetype，退回显示英文（不支持中文）
            cv2.putText(frame, name, (left + 2, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame


def face_show():
    # 加载人脸数据库
    known_face_encodings, known_face_names = load_face_database()

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # 尝试加载 freetype 字体模块
    ft_renderer = None
    try:
        ft_renderer = cv2.freetype.createFreeType2()
        ft_renderer.loadFontData(fontFileName="simhei.ttf", id=0)
    except Exception as e:
        print("Warning: 无法加载 simhei.ttf 或 OpenCV FreeType 模块不可用。中文可能无法正常显示。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # 检测人脸信息
        face_info = detect_faces(frame, known_face_encodings, known_face_names)

        # 显示人脸信息
        frame_with_faces = display_faces(frame, face_info, ft_renderer)
        cv2.imshow('Face Recognition', frame_with_faces)

        # 按Q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
