from .config_ActionDetector import ActionDetectorConfig
from .network import PoseCNN
from .dataset import PoseDataset
from .utils import mp_pose, pose, mp_drawing
import torch


def load_model(my_config=None):
    if my_config is None:
        my_config = ActionDetectorConfig()
    labels = my_config.labels

    model = PoseCNN(num_classes=len(labels))
    model.load_state_dict(torch.load(my_config.model_path, map_location='cpu'))
    model.eval()
    return model


__all__ = [
    'PoseCNN', 'PoseDataset',
    'load_model', 'mp_pose',
    'pose', 'mp_drawing'
]
