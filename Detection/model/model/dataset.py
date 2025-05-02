import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PoseDataset(Dataset):
    def __init__(self, base_path, label_map):
        self.data = []
        self.labels = []
        for label in label_map:
            folder = os.path.join(base_path, label)
            for file in os.listdir(folder):
                if file.endswith('.txt'):
                    filepath = os.path.join(folder, file)
                    coords = np.loadtxt(filepath, usecols=(1, 2, 3, 4))  # 只取x, y, z, score
                    if coords.shape[0] == 33:  # 确保有33个点
                        self.data.append(coords)
                        self.labels.append(label_map[label])
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # shape: (33, 4)
        x = x.T  # shape: (4, 33) for CNN input
        return torch.tensor(x), torch.tensor(self.labels[idx])