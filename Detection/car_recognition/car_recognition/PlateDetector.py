import cv2
import numpy as np
from .PlateNeuralNet import PlateCnnNet
from .CharNeuralNet import CharCnnNet
from scipy.signal import argrelextrema
from itertools import combinations
from PIL import ImageFont, ImageDraw, Image
import os

'''
车牌位置信息：
    'center': 车牌中心位置,
    'width': 车牌宽度,
    'height': 车牌高度,
    'angle': 车牌角度,
    'vertices': 车牌顶点信息,
    'cropped': 裁剪扭正后车牌二值图片,
    'chars': 列表，每一个元素char为分割后的单个字符图片,
    'plate_number': 车牌识别后的信息
'''


# 确认一下车牌位置
def locate_plate(img, debug=False):
    """
    定位图像中的车牌区域

    :param img: 输入图片（视频单帧）
    :param debug: 是否在图像上绘制调试信息
    :return: 初步筛选的可能车牌位置信息，以及原图（可选带标注）
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 蓝色车牌的HSV范围
    lower_blue = np.array([100, 80, 80])
    upper_blue = np.array([130, 255, 255])

    # 黄色车牌的HSV范围
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([40, 255, 255])

    # 创建掩码
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(mask_blue, mask_yellow)

    # 边缘检测
    edges = cv2.Canny(mask, 100, 200)

    # 膨胀 + 闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plates_info = []

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (center_x, center_y), (width, height), angle = rect

        if width == 0 or height == 0:
            continue

        ratio = max(width, height) / min(width, height)

        if 2 < ratio < 6:
            if width < height:
                angle += 90
                width, height = height, width

            box = cv2.boxPoints(rect)
            box = box.astype(int)

            plates_info.append({
                'center': (round(center_x, 2), round(center_y, 2)),
                'width': round(width, 2),
                'height': round(height, 2),
                'angle': round(angle, 2),
                'vertices': [tuple(map(int, point)) for point in box]
            })

            if debug:
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                cx, cy = int(center_x), int(center_y)
                cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)

    return img, plates_info


# 旋转扭正
def rotate_and_crop_plates(img, plates_info):
    """
    旋转并裁剪车牌图像，将裁剪结果添加到每个 plate 信息中

    :param img: 原始图像
    :param plates_info: 包含每个候选车牌位置和角度信息的字典列表
    :return: 更新后的 plates_info，其中每个 plate 字典包含键 'cropped'
    """
    updated_plates_info = []

    for plate in plates_info:
        center = plate['center']
        width = plate['width']
        height = plate['height']
        angle = plate['angle']

        if height > width:
            width, height = height, width
            angle += 90

        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        center_int = tuple(map(int, center))
        size = (int(width), int(height))

        rot_mat = cv2.getRotationMatrix2D(center_int, angle, 1.0)
        rotated_img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))

        x, y = center_int
        w, h = size
        x1 = max(0, x - w // 2)
        y1 = max(0, y - h // 2)
        x2 = min(rotated_img.shape[1], x + w // 2)
        y2 = min(rotated_img.shape[0], y + h // 2)

        if x2 <= x1 or y2 <= y1:
            continue

        cropped_plate = rotated_img[y1:y2, x1:x2]

        # 添加裁剪结果到字典
        plate['cropped'] = cropped_plate
        updated_plates_info.append(plate)

    return img, updated_plates_info


# 过滤不符合的
def plate_filter(plates_info, model_path, device=None):
    """
    使用 CNN 模型过滤掉不合格的车牌

    :param plates_info: 包含 'cropped' 图像字段的 plate 字典列表
    :param model_path: CNN 模型路径
    :param device: 推理设备（如 'cpu' 或 'cuda'）
    :return: 筛选后的 plate 字典列表（仅保留分类通过的）
    """
    net = PlateCnnNet()
    filtered_plates = []

    for plate in plates_info:
        cropped = plate.get('cropped')
        if cropped is None or cropped.size == 0:
            continue
        if net.test(cropped, model_path, device):
            filtered_plates.append(plate)

    return filtered_plates


# 去掉上下不必要部分
def find_waves(threshold, histogram):
    """ 根据设定的阈值和图片直方图，找出波峰，用于分隔字符 """
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    i = 0
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


def remove_up_down_border(plates_info):
    """ 去除每张车牌图像上下边缘，并将处理结果写回 plate['cropped'] """
    for plate in plates_info:
        img = plate.get('cropped')
        if img is None:
            print("警告：未找到 'cropped' 图像，跳过")
            continue

        if len(img.shape) == 3 and img.shape[2] == 3:
            plate_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            plate_gray = img  # 已经是灰度图

        _, binary = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        row_hist = np.sum(binary, axis=1)
        row_min = np.min(row_hist)
        row_avg = np.mean(row_hist)
        threshold = (row_min + row_avg) / 2

        wave_peaks = find_waves(threshold, row_hist)
        if not wave_peaks:
            print("警告：未检测到有效的上下边界波峰，跳过")
            continue

        selected_wave = max(wave_peaks, key=lambda x: x[1] - x[0])
        y1, y2 = selected_wave[0], selected_wave[1]
        plate['cropped'] = binary[y1:y2, :]  # 覆盖写入原始字典
        print("上下边界去除成功")

    return plates_info  # 可选返回


# 去掉左右两边噪声
def crop_plate_region(plates_info, margin=2):
    """
    移除车牌左右白边噪声，修改 plates_info 中的 cropped 信息。
    :param plates_info: 每个 plate 包含了 'cropped' 图像字段的字典列表
    :param margin: 左右边界的空余像素点
    :return: 修改后的 plates_info
    """
    for plate in plates_info:
        img = plate.get('cropped')
        if img is None:
            continue  # 如果没有 'cropped' 图像，跳过

        col_sum = np.sum(img, axis=0)
        threshold = np.max(col_sum) * 0.1
        left, right = 0, img.shape[1] - 1

        # 向右寻找第一个大于阈值的列
        while left < right and col_sum[left] < threshold:
            left += 1
        # 向左寻找第一个大于阈值的列
        while right > left and col_sum[right] < threshold:
            right -= 1

        # 更新 cropped 信息
        plate['cropped'] = img[:, max(0, left - margin):min(img.shape[1], right + margin)]
        print("左右边界去除成功")

    return plates_info  # 返回修改后的 plates_info


# 字符分割
def split_license_plate_by_valley(plates_info, debug=False):
    """
    对每个车牌图像进行 valley 分割，结果写入 plate['chars']。
    输入和输出都是 plates_info。
    """
    for plate in plates_info:
        img = plate.get('cropped')
        if img is None:
            plate['chars'] = []
            continue

        # 处理图像
        # 包装为单元素列表以便复用已有函数
        plate = remove_up_down_border([plate])[0]
        plate = crop_plate_region([plate])[0]
        img = plate['cropped']

        if img is None or img.shape[1] < 10:
            print("警告：图像处理后尺寸过小，跳过字符分割")
            plate['chars'] = []
            continue

        col_sum = np.sum(img, axis=0)
        win_size = max(3, img.shape[1] // 40)
        kernel = np.ones(win_size) / win_size
        cov = np.convolve(col_sum, kernel, mode='same')

        valley_centers = find_valleys(cov, num_valleys=6)
        split_points = [0] + valley_centers + [img.shape[1]]

        if len(split_points) < 2:
            print("无法从图像中分割字符")
            plate['chars'] = []
            continue

        chars = []
        for i in range(len(split_points) - 1):
            ch = img[:, split_points[i]:split_points[i + 1]]
            if ch.shape[1] > 2:  # 避免空字符或太窄的区域
                chars.append(pad_image_to_square(ch))

        plate['chars'] = chars

        if debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 3))
            plt.plot(cov)
            for x in valley_centers:
                plt.axvline(x, color='r', linestyle='--')
            plt.title('Valley-based Splits (with flat zones)')
            plt.show()

    return plates_info


# 填充避免字符识别resize识别出错
def pad_image_to_square(img):
    """
    对图像左右填充黑色像素，使其变为正方形（宽度 = 高度）。
    仅当宽度 < 高度时进行填充。
    """
    h, w = img.shape
    if w >= h:
        return img  # 无需填充

    pad_total = h - w
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    padded_img = np.pad(img, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
    return padded_img


# 找谷底
def find_valleys(cov, num_valleys=6, kernel_size1=5, kernel_size2=3, times=3):
    """
    在平滑后的列投影中查找 valley（极小值），
    并选择使分割后宽度标准差最小的6个 valley 分割点。
    如果 valley 不足，会回退使用均匀分布的点填补。
    """
    # Step 1: 平滑处理
    kernel = np.ones(kernel_size1) / kernel_size1
    smoothed = np.convolve(cov, kernel, mode='same')
    for _ in range(times):
        kernel = np.ones(kernel_size2) / kernel_size2
        smoothed = np.convolve(smoothed, kernel, mode='same')

    # Step 2: 寻找极小值点
    local_minima = argrelextrema(smoothed, np.less)[0].tolist()
    global_min = int(np.argmin(smoothed))
    if global_min not in local_minima:
        local_minima.append(global_min)

    valley_candidates = sorted(set(local_minima))

    # Step 3: 找最优组合（均方差最小）
    best_comb = None
    best_std = float('inf')
    for comb in combinations(valley_candidates, min(len(valley_candidates), num_valleys)):
        splits = [0] + list(comb) + [len(cov)]
        widths = [splits[i+1] - splits[i] for i in range(len(splits)-1)]
        std = np.std(widths)
        if std < best_std:
            best_std = std
            best_comb = list(comb)

    # Step 4: 不足时，补充 valley（使用均匀切分）
    if best_comb is None or len(best_comb) < num_valleys:
        print("⚠️ Valley 不足，使用均匀分布填补")
        est = np.linspace(0, len(cov), num_valleys + 2)[1:-1]
        return [int(x) for x in est]

    return sorted(best_comb)


# 识别字符
def char_recognition(plates_info, model_path):
    # 网络
    char_cnn_net = CharCnnNet()
    for plate in plates_info:
        # 获取chars
        chars = plate.get('chars')

        if not chars:
            plate['plate_number'] = ''
            continue

        plate_number = ''
        for char in chars:
            if char is None or char.shape[1] < 2:
                print("警告：字符图像无效，跳过")
                continue

            black_pixels = np.sum(char == 0)
            white_pixels = np.sum(char == 255)
            print("black:", black_pixels)
            print("white:", white_pixels)
            # 判断是否为白底黑字（白色像素多于黑色）
            if white_pixels > black_pixels:
                char = 255 - char
                print("检测为白底黑字，已翻转。")
            else:
                print("检测为黑底白字，无需翻转。")

            pred = char_cnn_net.test(char, model_path)
            plate_number += pred

        plate['plate_number'] = plate_number

    return plates_info


# 识别总流程
def plate_detector(img, plate_model_path, char_model_path,
                   debug_locate=False, debug_split=False):
    # 获取色块位置信息
    img, plates_info = locate_plate(img, debug_locate)
    # 将其扭正放入plate info中
    cropped_aligned_plates, plates_info = rotate_and_crop_plates(img, plates_info)
    # 过滤得到车牌位置
    plates_info = plate_filter(plates_info, model_path=plate_model_path)
    # 分割字符
    split_license_plate_by_valley(plates_info, debug=debug_split)
    # 车牌字符识别
    char_recognition(plates_info, char_model_path)
    return plates_info


def plate_detector_with_overlay(
    img, plate_model_path, char_model_path,
    debug_locate=False, debug_split=False,
    font_path='C:/Windows/Fonts/simhei.ttf',
    box_color=(255, 0, 0), box_thickness=5,
    font_size=30, font_color=(255, 0, 0)
):
    """
    识别图像中的车牌并在图像上绘制车牌边框和识别结果（支持中文）

    :param img: 输入图像（BGR格式）
    :param plate_model_path: 车牌分类模型路径
    :param char_model_path: 字符识别模型路径
    :param debug_locate: 是否调试车牌定位
    :param debug_split: 是否调试字符切割
    :param font_path: 中文字体路径
    :param box_color: 边框颜色（R, G, B）
    :param box_thickness: 边框线宽
    :param font_size: 字体大小
    :param font_color: 字体颜色（R, G, B）
    :return: 标注图像、副本的 plates_info 列表
    """
    draw_img = img.copy()

    # 识别流程
    _, plates_info = locate_plate(draw_img, debug=debug_locate)
    _, plates_info = rotate_and_crop_plates(draw_img, plates_info)
    plates_info = plate_filter(plates_info, model_path=plate_model_path)
    split_license_plate_by_valley(plates_info, debug=debug_split)
    char_recognition(plates_info, char_model_path)

    # 加载字体
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"字体文件未找到：{font_path}")
    font = ImageFont.truetype(font_path, font_size)

    # PIL 图像绘制
    pil_img = Image.fromarray(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for plate in plates_info:
        vertices = plate.get('vertices')
        plate_number = plate.get('plate_number', '')

        if vertices and len(vertices) == 4:
            # 绘制四边形边框
            polygon = [tuple(pt) for pt in vertices] + [tuple(vertices[0])]
            draw.line(polygon, fill=box_color, width=box_thickness)

            # 文字位置
            top_center = tuple(np.mean(vertices[:2], axis=0).astype(int))
            text_pos = (top_center[0], max(0, top_center[1] - font_size - 4))
            draw.text(text_pos, plate_number, font=font, fill=font_color)

    # 转回 OpenCV 图像
    result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return result_img, plates_info
