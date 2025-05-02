import torch.nn as nn
import torch.nn.functional as F


# drop out
class PoseCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

# class PoseCNN(nn.Module):
#     def __init__(self, num_classes=5):
#         super(PoseCNN, self).__init__()
#
#         self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(64, num_classes)
#
#     def forward(self, x):  # x shape: (batch_size, 4, 33)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.pool(x).squeeze(-1)  # shape: (batch_size, 64)
#         x = self.fc(x)
#         return x