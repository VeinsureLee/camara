import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
import cv2
from . import load_model, ActionDetectorConfig


def load_pose_txt(txt_path):
    coords = np.loadtxt(txt_path, usecols=(1, 2, 3, 4))  # 只取 x, y, z, score
    if coords.shape[0] != 33:
        raise ValueError(f"点数量不是33个，实际为 {coords.shape[0]} 个点")
    coords = coords.T  # (4, 33)
    coords = np.expand_dims(coords, axis=0)  # (1, 4, 33) for batch size 1
    return torch.tensor(coords, dtype=torch.float32)


def load_pose_from_image(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像：{image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        raise ValueError("未检测到人体姿态")

    landmarks = results.pose_landmarks.landmark
    if len(landmarks) != 33:
        raise ValueError(f"检测到的点数不是33个，而是 {len(landmarks)}")

    coords = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])  # (33, 4)
    coords = coords.T  # 转置为 (4, 33)
    coords = np.expand_dims(coords, axis=0)  # 添加 batch 维度，(1, 4, 33)

    return torch.tensor(coords, dtype=torch.float32)


def pose_status(image, config=ActionDetectorConfig()):
    model = load_model(config)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    if image is None:
        raise ValueError("输入图像为 None")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return 'No action detected'

    landmarks = results.pose_landmarks.landmark
    if len(landmarks) != 33:
        raise ValueError(f"检测到的点数不是33个，而是 {len(landmarks)}")

    coords = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])  # (33, 4)
    coords = coords.T  # 转置为 (4, 33)
    coords = np.expand_dims(coords, axis=0)  # (1, 4, 33)
    coords_tensor = torch.tensor(coords, dtype=torch.float32)

    with torch.no_grad():
        output = model(coords_tensor)  # 用处理后的坐标输入模型
        pred = torch.argmax(output, dim=1).item()

    return config.labels[pred]


def predict(config=ActionDetectorConfig()):
    model = load_model(config)
    labels = config.labels
    # 示例：预测 ./test/001.txt
    input_tensor = load_pose_txt('data/coordinates/test/fall02.txt')
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        print(f"预测类别: {labels[pred]}")

    input_img_tensor = load_pose_from_image('data/photo/test/fall02.png')
    with torch.no_grad():
        output = model(input_img_tensor)
        pred = torch.argmax(output, dim=1).item()
        print(f"预测类别: {labels[pred]}")


