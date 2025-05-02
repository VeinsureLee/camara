import sys
import os
import numpy as np
import cv2
import tensorflow as tf  # 虽然导入了 tensorflow，但实际未使用
from sklearn.model_selection import train_test_split
from . import numbers, alphabets, chinese  # 未在本文件中使用，可能为通用模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from .config_CarDetector import CarDetectorConfig
import time


class PlateCnnNet(nn.Module):
    """
    基于 PyTorch 的车牌存在性检测 CNN 网络
    输入一张车牌区域图像，输出是否包含车牌（2 类：有/无）
    """
    def __init__(self):
        super(PlateCnnNet, self).__init__()
        # 输入图像宽、高
        self.img_w, self.img_h = 136, 36
        # 二分类输出
        self.y_size = 2
        # 训练批大小
        self.batch_size = 100
        # 学习率默认值
        self.learn_rate = 0.001
        # dropout 保留概率
        self.keep_prob = 0.5

        # 卷积层 1: 输入 3 通道 -> 32 通道, 卷积核 3x3, padding=1 保持尺寸
        # 池化后，尺寸减半: 136x36 -> 68x18
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层 2: 32 -> 64, 池化后: 68x18 -> 34x9
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层 3: 64 -> 128, 池化: 34x9 -> 17x4
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算经过 3 次 2x2 池化后的特征图展平维度
        conv_h = self.img_h // 8  # 36/8 = 4
        conv_w = self.img_w // 8  # 136/8 = 17
        self.flat_dim = conv_h * conv_w * 128

        # 全连接层定义
        self.fc1 = nn.Linear(self.flat_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, self.y_size)

        # dropout 层
        self.dropout = nn.Dropout(p=self.keep_prob)

    def forward(self, x):
        """
        前向计算
        输入 x: (batch_size, 3, img_h, img_w)
        返回: (batch_size, 2)
        """
        # 卷积+ReLU+池化+dropout
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout(x)

        # 展平
        x = x.view(-1, self.flat_dim)

        # 全连接 + ReLU + dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # 输出层, 未加激活, 与 CrossEntropyLoss 配合使用
        out = self.fc3(x)
        return out

    def list_all_files(self, root):
        """
        递归遍历目录, 返回所有文件路径
        """
        files = []
        names = os.listdir(root)
        for name in names:
            path = os.path.join(root, name)
            if os.path.isdir(path):
                files.extend(self.list_all_files(path))
            elif os.path.isfile(path):
                files.append(path)
        return files

    def init_data(self, dir):
        """
        初始化训练数据：读取图像并生成标签
        目录下子目录名为 'has' 或其他表示无车牌
        返回 X (N, H, W, C), y (N, 2)
        """
        X, y = [], []
        if not os.path.exists(dir):
            raise ValueError('没有找到文件夹')
        files = self.list_all_files(dir)
        # 根据父目录名称生成标签列表
        labels = [os.path.basename(os.path.dirname(f)) for f in files]

        for idx, file in enumerate(files):
            img = cv2.imread(file)
            # 跳过非彩色图
            if img is None or img.ndim != 3:
                continue
            # 调整尺寸到 (W, H)
            resized = cv2.resize(img, (self.img_w, self.img_h))
            X.append(resized)
            # 有车牌 -> [0,1], 无车牌 -> [1,0]
            y.append([0, 1] if labels[idx] == 'has' else [1, 0])

        X = np.array(X)
        y = np.array(y).reshape(-1, 2)
        return X, y

    def init_testData(self, dir):
        """
        初始化测试数据：读取灰度图并转换为彩色三通道后 resize
        返回 test_X (M, H, W, C)
        """
        test_X = []
        if not os.path.exists(dir):
            raise ValueError('没有找到文件夹')
        files = self.list_all_files(dir)
        for file in files:
            img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
            # 跳过非灰度图
            if img is None or img.ndim != 2:
                continue
            # 转为 BGR 三通道
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            resized = cv2.resize(img_color, (self.img_w, self.img_h))
            test_X.append(resized)
        return np.array(test_X)

    def train_net(self, data_dir, num_epochs=10, lr=0.001, save_path='plate_cnn_model.pth', device=None):
        """
        训练网络并保存模型
        """
        # 选择设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        self.to(device)
        self.train()

        # 加载并预处理数据
        X, y = self.init_data(data_dir)
        X = X.astype(np.float32) / 255.0    # 归一化到 [0,1]
        # HWC -> CHW
        X = np.transpose(X, (0, 3, 1, 2))
        y = y.astype(np.float32)

        # 划分训练/验证集
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        # 转张量
        X_train_t = torch.tensor(X_train)
        y_train_t = torch.tensor(y_train)
        X_val_t = torch.tensor(X_val)
        y_val_t = torch.tensor(y_val)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=self.batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        print("开始训练...")
        for epoch in range(num_epochs):
            self.train()
            total_loss, correct, total = 0, 0, 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self(batch_X)
                labels = torch.argmax(batch_y, dim=1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}, Accuracy: {correct/total:.4f}")

            # 验证阶段
            self.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for vX, vy in val_loader:
                    vX, vy = vX.to(device), vy.to(device)
                    vout = self(vX)
                    vlabels = torch.argmax(vy, dim=1)
                    vpreds = torch.argmax(vout, dim=1)
                    val_correct += (vpreds == vlabels).sum().item()
                    val_total += vlabels.size(0)
            print(f"Validation Accuracy: {val_correct/val_total:.4f}")

        # 保存模型
        torch.save(self.state_dict(), save_path)
        print(f"模型保存至：{save_path}")

    def test(self, img, model_path, device=None):
        """
        加载模型并对单张图像进行推断，打印加载时间
        返回预测类别索引
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("正在加载模型...")
        t0 = time.time()
        self.load_state_dict(torch.load(model_path, map_location=device))
        self.to(device)
        self.eval()
        t1 = time.time()
        print(f"模型加载成功, 用时 {t1 - t0:.3f} 秒")

        # 预处理输入图像
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(img, (self.img_w, self.img_h))
        normed = resized.astype(np.float32) / 255.0
        transposed = np.transpose(normed, (2, 0, 1))
        tensor = torch.tensor(transposed).unsqueeze(0).to(device)

        with torch.no_grad():
            out = self(tensor)
            pred = torch.argmax(out, dim=1).item()
        return pred


def main(config=CarDetectorConfig()):
    """
    主函数，根据配置执行训练或测试流程
    """
    cur_dir = sys.path[0]
    data_dir = os.path.join(cur_dir, config.data_plate_dir)
    test_dir = os.path.join(cur_dir, config.test_plate_dir)
    train_path = os.path.join(cur_dir, config.train_model_plate_path)
    model_path = os.path.join(cur_dir, config.model_plate_path)

    train_flag = 0  # 1=训练, 0=测试
    net = PlateCnnNet()

    if train_flag == 1:
        net.train_net(data_dir, num_epochs=15, lr=1e-3, save_path=train_path)
    else:
        img = cv2.imread(test_dir)
        pred = net.test(img, model_path=model_path)
        print("预测结果:", pred)
