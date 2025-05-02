from ultralytics import YOLO
from .config_HumanDetector import HumanDetectorConfig


def load_model(my_config=HumanDetectorConfig()):

    model = YOLO(my_config.model_path)
    return model