from Detection.car_recognition.car_recognition.PlateDetector import plate_detector_with_overlay
from Detection.car_recognition.car_recognition.config_CarDetector import CarDetectorConfig
import cv2


def process_video(video_path, output_path, plate_model_path, char_model_path, font_path='C:/Windows/Fonts/simhei.ttf'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频基本信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化视频写入器（可选）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 处理每一帧
        result_frame, plates_info = plate_detector_with_overlay(
            frame,
            plate_model_path,
            char_model_path,
            font_path=font_path,
            debug_locate=False
        )

        out.write(result_frame)  # 保存帧（可选）

        # 如果想实时显示：
        cv2.imshow("Plate Detection", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        print(f"处理帧数: {frame_count}, 车牌数量: {len(plates_info)}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()


video_test_path = '../../demo/mp4/test01.mp4'
output_test_path = '../../demo/mp4/output_video01.mp4'
config = CarDetectorConfig()

process_video(video_test_path, output_test_path, config.plate_model_path, config.char_model_path)
