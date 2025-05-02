import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from . import numbers, alphabets, chinese  # 导入字符集：数字、字母、汉字
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from .config_CarDetector import CarDetectorConfig


class CharCnnNet(nn.Module):
    """
    基于 PyTorch 的字符分类 CNN 网络
    用于对输入的单字符图像进行分类，支持数字、字母和汉字
    """
    def __init__(self):
        super(CharCnnNet, self).__init__()
        # 构建字符集及其长度
        self.dataset = numbers + alphabets + chinese
        self.dataset_len = len(self.dataset)
        # 输入图像尺寸（20x20）
        self.img_size = 20
        # 输出类别数
        self.y_size = len(self.dataset)
        # 训练时的批大小
        self.batch_size = 100
        # dropout 保留概率
        self.keep_prob = 0.5

        # 定义卷积层和池化层
        # Conv1: 输入通道1 -> 输出通道32, 卷积核3x3, padding=1 保持尺寸
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化降采样 2x2

        # Conv2: 32 -> 64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv3: 64 -> 128
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层: 注意输入维度 = 上一层特征图展平后尺寸 2*2*128
        self.fc1 = nn.Linear(2 * 2 * 128, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, self.y_size)  # 输出到类别数

        # dropout 层
        self.dropout = nn.Dropout(p=self.keep_prob)

    def forward(self, x=None):
        # x 形状 (B, 1, 20, 20)
        x = x.view(-1, 1, self.img_size, self.img_size)

        # 第一卷积块 + ReLU + 池化 + dropout
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.dropout(x, p=self.keep_prob, training=self.training)

        # 第二卷积块
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.dropout(x, p=self.keep_prob, training=self.training)

        # 第三卷积块
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.dropout(x, p=self.keep_prob, training=self.training)

        # 展平特征图
        x = x.view(-1, 2 * 2 * 128)
        # 全连接层 + ReLU + dropout
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.keep_prob, training=self.training)

        # 第二个全连接层
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.keep_prob, training=self.training)

        # 输出层, 不使用激活, 后续与 CrossEntropyLoss 一起使用
        x = self.fc3(x)
        return x

    def list_all_files(self, root):
        """
        递归遍历 root 目录, 返回所有文件路径列表
        仅进入目录名在 dataset 中的子目录
        """
        files = []
        list_files = os.listdir(root)
        for element_name in list_files:
            element = os.path.join(root, element_name)
            if os.path.isdir(element):
                # 如果目录名是字符集中的一类, 则继续遍历
                if element_name in self.dataset:
                    files.extend(self.list_all_files(element))
            elif os.path.isfile(element):
                files.append(element)
        return files

    def init_data(self, data_dir):
        """
        读取训练/验证数据目录下所有字符图像, 返回 features, labels
        features: (N, 20, 20) 灰度图数组
        labels: one-hot 编码 (N, dataset_len)
        """
        features, labels = [], []
        if not os.path.exists(data_dir):
            raise ValueError('没有找到文件夹')
        files = self.list_all_files(data_dir)

        for file in files:
            # 以灰度模式读取
            src_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            # 跳过非单通道图像
            if src_img.ndim == 3:
                continue
            # 缩放到 20x20
            resize_img = cv2.resize(src_img, (self.img_size, self.img_size))
            features.append(resize_img)
            # 获取文件上一级目录名作为标签
            dir_name = os.path.basename(os.path.dirname(file))
            # 构造 one-hot 向量
            vector_y = [0] * self.dataset_len
            idx = self.dataset.index(dir_name)
            vector_y[idx] = 1
            labels.append(vector_y)

        # 转为 numpy 数组并返回
        features = np.array(features)
        labels = np.array(labels).reshape(-1, self.dataset_len)
        return features, labels

    def init_testdata(self, test_dir):
        """
        读取测试目录下图像, 返回 numpy 数组 (M, 20, 20)
        """
        test_X = []
        if not os.path.exists(test_dir):
            raise ValueError('没有找到文件夹')
        files = self.list_all_files(test_dir)
        for file in files:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if img.ndim == 3:
                continue
            resize_img = cv2.resize(img, (self.img_size, self.img_size))
            test_X.append(resize_img)
        return np.array(test_X)

    def train_net(self, data_dir, num_epochs=10, lr=0.001, save_path='plate_cnn_model.pth', device=None):
        """
        训练网络, 并保存模型
        data_dir: 训练数据目录
        num_epochs: 训练轮数
        lr: 学习率
        save_path: 模型保存路径
        device: CPU 或 GPU
        """
        # 设备选择: 优先 GPU
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        self.to(device)
        self.train()

        # 加载数据并归一化
        X, y = self.init_data(data_dir)
        X = X.astype(np.float32) / 255.0
        y = y.astype(np.float32)

        # 划分训练/验证集
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        # 转为 Tensor 并添加通道维度
        X_train_tensor = torch.tensor(X_train).unsqueeze(1)
        y_train_tensor = torch.tensor(y_train)
        X_val_tensor = torch.tensor(X_val).unsqueeze(1)
        y_val_tensor = torch.tensor(y_val)

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=self.batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        print("开始训练...")
        for epoch in range(num_epochs):
            self.train()
            total_loss, correct, total = 0, 0, 0

            # 训练循环
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

            acc = correct / total
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

            # 验证阶段
            self.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for val_X, val_y in val_loader:
                    val_X, val_y = val_X.to(device), val_y.to(device)
                    val_outputs = self(val_X)
                    val_labels = torch.argmax(val_y, dim=1)
                    val_preds = torch.argmax(val_outputs, dim=1)
                    val_correct += (val_preds == val_labels).sum().item()
                    val_total += val_labels.size(0)
            val_acc = val_correct / val_total
            print(f"Validation Accuracy: {val_acc:.4f}")

        # 保存训练好的模型参数
        torch.save(self.state_dict(), save_path)
        print(f"模型保存至：{save_path}")

    def test(self, img, model_path, device=None):
        """
        使用训练好的模型对单张图像进行预测
        img: 输入图像 (灰度或 BGR)
        model_path: 模型文件路径
        返回: 预测的字符
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型参数
        self.load_state_dict(torch.load(model_path, map_location=device))
        self.to(device)
        self.eval()

        # 如果为彩色, 转灰度
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 预处理: 缩放+归一化+转 Tensor
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self(img_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            predicted_char = self.dataset[pred_idx]
            return predicted_char


def main(config=CarDetectorConfig()):
    """
    主函数: 根据配置决定训练或测试
    """
    cur_dir = sys.path[0]
    data_dir = os.path.join(cur_dir, config.data_char_dir)
    test_dir = os.path.join(cur_dir, config.test_char_dir)
    train_model_path = os.path.join(cur_dir, config.train_model_char_path)
    model_path = os.path.join(cur_dir, config.model_char_path)

    train_flag = 0  # 训练标志: 1=训练, 0=测试
    net = CharCnnNet()

    if train_flag == 1:
        # 执行训练
        net.train_net(data_dir, num_epochs=15, lr=1e-3, save_path=train_model_path)
    else:
        # 执行测试: 读取测试图像, 并打印预测结果
        img = cv2.imread(test_dir, cv2.IMREAD_GRAYSCALE)
        pred_char = net.test(img, model_path=model_path)
        print("预测结果:", pred_char)
