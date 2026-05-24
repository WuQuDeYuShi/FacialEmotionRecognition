import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import os

# ------------------------- 1. 超参数设置 -------------------------
BATCH_SIZE = 64          # 批大小
EPOCHS = 30              # 训练轮数
LEARNING_RATE = 0.001    # 学习率
NUM_CLASSES = 7          # 情绪类别数（生气、厌恶、恐惧、高兴、悲伤、惊讶、中性）
IMG_SIZE = 48            # 输入图像尺寸（FER2013 常用 48x48）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------- 2. 数据预处理 -------------------------
# 训练集数据增强与归一化
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),          # 转为单通道灰度图
    transforms.Resize((IMG_SIZE, IMG_SIZE)),              # 统一尺寸
    transforms.RandomHorizontalFlip(p=0.5),               # 随机水平翻转，增强泛化
    transforms.RandomRotation(degrees=10),                # 随机旋转 ±10°
    transforms.ToTensor(),                                # 转为 Tensor 并缩放到 [0,1]
    transforms.Normalize(mean=[0.5], std=[0.5])           # 归一化到 [-1,1]（灰度图通道数=1）
])

# 验证/测试集仅做基本预处理
val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ------------------------- 3. 加载数据集（假设按文件夹组织）-------------------------
# 数据目录结构示例：
# data/
#   train/
#       angry/
#       disgust/
#       ...
#   val/
#       angry/
#       ...
def load_data(data_root):
    """
    使用 ImageFolder 加载数据，并按 8:2 划分训练集和验证集（若没有单独的 val 文件夹）
    :param data_root: 数据根目录，包含 train 和 val 子目录（推荐）或只含 train
    :return: train_loader, val_loader
    """
    train_path = os.path.join(data_root, 'train')
    val_path = os.path.join(data_root, 'val')
    
    if os.path.exists(val_path):
        # 已有独立验证集
        train_dataset = ImageFolder(train_path, transform=train_transform)
        val_dataset = ImageFolder(val_path, transform=val_transform)
    else:
        # 从训练集中划分 20% 作为验证集
        full_dataset = ImageFolder(train_path, transform=train_transform)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        # 注意：random_split 返回的子集无法直接获取类别，但可通过 dataset.dataset.classes 获取原始类别名
        # 为方便，将 val_dataset 的 transform 改为 val_transform（需重新包装）
        val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, val_loader

# ------------------------- 4. 定义 CNN 模型 -------------------------
class EmotionCNN(nn.Module):
    """
    情绪识别卷积神经网络：
    - 3个卷积块（卷积+批归一化+ReLU+池化）
    - 2个全连接层
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(EmotionCNN, self).__init__()
        # 卷积块1：输入 1x48x48 -> 32x24x24
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 尺寸减半
        )
        # 卷积块2：32x24x24 -> 64x12x12
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # 卷积块3：64x12x12 -> 128x6x6
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # 全局平均池化替代全连接，减少参数量（也可保留展平）
        # 此处采用展平方式：
        self.flatten = nn.Flatten()
        # 计算展平后的特征维度：128 * 6 * 6 = 4608
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ------------------------- 5. 训练与验证函数 -------------------------
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一个 epoch
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """
    验证模型性能
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    return val_loss, val_acc, all_preds, all_labels

# ------------------------- 6. 主训练流程 -------------------------
def main():
    # 加载数据（请替换为实际数据路径）
    data_root = "./data/emotion"   # 修改为你的数据路径
    train_loader, val_loader = load_data(data_root)
    
    # 实例化模型、损失函数、优化器
    model = EmotionCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()   # 多分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 记录训练过程
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)   # 根据验证损失调整学习率
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_emotion_cnn.pth")
            print(f"  --> 保存最佳模型，验证准确率: {val_acc:.4f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.show()
    
    # 最终评估并绘制混淆矩阵
    _, _, preds, labels = validate(model, val_loader, criterion, DEVICE)
    cm = confusion_matrix(labels, preds)
    class_names = val_loader.dataset.dataset.classes if hasattr(val_loader.dataset, 'dataset') else val_loader.dataset.classes
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

# ------------------------- 7. 单张图片预测示例（可选）-------------------------
def predict_image(model_path, image_path, device=DEVICE):
    """
    加载训练好的模型，对单张人脸图像进行情绪预测
    """
    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    from PIL import Image
    image = Image.open(image_path).convert('RGB')   # 确保彩色图转为灰度时模式正确
    image = transform(image).unsqueeze(0)           # 增加 batch 维度
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        prob = F.softmax(output, dim=1)
        pred_class = torch.argmax(prob, dim=1).item()
    
    # 情绪标签顺序需要与训练时一致（通常为：0=生气,1=厌恶,2=恐惧,3=高兴,4=悲伤,5=惊讶,6=中性）
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    return emotions[pred_class], prob.cpu().numpy()

if __name__ == "__main__":
    main()
    # 单图预测示例（取消注释并修改路径）
    # pred, prob = predict_image("best_emotion_cnn.pth", "test_face.jpg")
    # print(f"预测情绪: {pred}, 概率分布: {prob}")