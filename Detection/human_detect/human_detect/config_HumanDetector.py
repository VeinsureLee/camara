from ultralytics import YOLO


class HumanDetectorConfig:
    def __init__(
            self,
            img_path="img_test/test01.jpg",
            model_path="model/yolov8n.pt",  # test data base save path
    ):
        self.img_path = img_path
        self.model_path = model_path


if __name__ == '__main__':
    # 默认配置
    default_config = HumanDetectorConfig()
