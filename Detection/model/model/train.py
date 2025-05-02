import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import logging
from . import PoseCNN, PoseDataset, ActionDetectorConfig

'''
依据已有训练数据集训练
每一次学习都将训练数据集按照8:2比例将其转化为train:val进行训练
'''


def train(config=ActionDetectorConfig()):
    logging.basicConfig(
        filename=config.train_log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Logging initialized. Log file: %s", config.train_log_path)
    base_path = config.base_path
    labels = config.labels
    num_epochs = config.num_epochs
    save_path = config.save_path
    logging.info("start training...")  # <-- 不需要再配置 basicConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseCNN(num_classes=len(labels)).to(device)

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    label_map = {label: idx for idx, label in enumerate(labels)}

    dataset = PoseDataset(base_path, label_map)
    best_acc = 0.0

    for epoch in range(num_epochs):
        train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, shuffle=True)
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
        logging.info(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Validation Accuracy: {acc:.2%}")
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Acc: {acc:.2%}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f"新最佳模型保存！准确率：{acc:.2%}")
            logging.info(f"new best model saved! acc: {acc:.2%}")

    print(f"最佳验证精度：{best_acc:.2%}")
    logging.info(f"Best Acc: {best_acc:.2f}")
