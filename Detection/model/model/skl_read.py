import os
import cv2
import mediapipe as mp
from . import mp_drawing, mp_pose, pose, load_model, ActionDetectorConfig

'''
读取root_input_folder下的照片并将33个点的数据其转化到root_output_folder里面
分别读取训练数据集和测试数据集（需要手动修改）
'''


def data_preparation(config=ActionDetectorConfig()):
    root_output_folder = [config.root_total_output_folder, config.root_test_output_folder]
    root_input_folder = [config.root_total_input_folder, config.root_test_input_folder]
    labels = config.labels
    for i in range(2):
        for label in labels:
            print(f"Processing {label}")
            input_folder = os.path.join(root_input_folder[i], label)
            output_folder = os.path.join(root_output_folder[i], label)
            os.makedirs(output_folder, exist_ok=True)

            deleted = 0
            # 遍历文件夹中的所有图片文件
            for filename in os.listdir(input_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(input_folder, filename)
                    image = cv2.imread(image_path)
                    if image is None:
                        deleted += 1
                        print(f"无法读取图片: {image_path}")
                        os.remove(image_path)
                        continue

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)

                    # 准备输出txt文件路径
                    txt_filename = os.path.splitext(filename)[0] + '.txt'
                    txt_path = os.path.join(output_folder, txt_filename)

                    if results.pose_landmarks:
                        h, w, _ = image.shape
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            for idx, lm in enumerate(results.pose_landmarks.landmark):
                                x, y, z = int(lm.x * w), int(lm.y * h), lm.z
                                f.write(f'{idx} {x} {y} {z:.4f} {lm.visibility:.4f}\n')
                    else:
                        deleted += 1
                        os.remove(image_path)
                        print(f"未检测到人体姿态: {filename}")

            print(f"在{label}已经删除{deleted}张图片")
