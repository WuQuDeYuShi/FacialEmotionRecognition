import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import os
import threading
from PIL import Image
from enum import Enum

# ------------------------- 1. 超参数设置 -------------------------
BATCH_SIZE = 64          # 批大小
EPOCHS = 30              # 训练轮数
LEARNING_RATE = 0.001    # 学习率
NUM_CLASSES = 7          # 情绪类别数（生气、厌恶、恐惧、高兴、悲伤、惊讶、中性）
IMG_SIZE = 48            # 输入图像尺寸（FER2013 常用 48x48）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------- 2. 数据预处理 -------------------------
class EmotionCSVDataset(Dataset):
    """
    从 CSV 文件读取人脸情绪数据的 Dataset
    Args:
        csv_path (str): CSV 文件路径
        transform (callable, optional): 图像预处理变换（与之前相同）
        usage_filter (str, optional): 如果 CSV 有 'Usage' 列，用于过滤子集（'Training', 'PublicTest' 等）
    """
    def __init__(self, csv_path, transform=None, usage_filter=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # 如果有 Usage 列，则按 usage_filter 过滤行
        if usage_filter is not None and 'Usage' in self.df.columns:
            self.df = self.df[self.df['Usage'] == usage_filter]
            self.df = self.df.reset_index(drop=True)
        
        # 确认必要的列存在
        assert 'emotion' in self.df.columns, "CSV 缺少 'emotion' 标签列"
        assert 'pixels' in self.df.columns, "CSV 缺少 'pixels' 像素列"
        
        # 预先解析像素为 numpy 数组，加快读取速度（若 CSV 很大，可改为在 __getitem__ 中解析）
        self.pixels = self.df['pixels'].apply(lambda x: np.fromstring(x, sep=' ', dtype=np.uint8)).values
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取标签
        label = self.df.iloc[idx]['emotion']
        
        # 获取像素字符串
        pixels = self.pixels[idx]                     # 直接取用，无需解析
        
        # 检查尺寸：48x48=2304，若是 64x64 则改为 4096
        img_size = int(np.sqrt(pixels.shape[0]))  # 自动推断宽高（假设正方形）
        pixels = pixels.reshape((img_size, img_size))
        
        # 转换为 PIL Image 对象，以便后续使用原有的 transform（包含灰度、缩放、旋转等）
        # 注意：pixels 已经是单通道灰度，模式用 'L'
        image = Image.fromarray(pixels, mode='L')
        
        # 应用预处理变换（包括 ToTensor 和 Normalize）
        if self.transform:
            image = self.transform(image)
        return image, label
    
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

# ------------------------- 3. 加载数据集（.csv文件）-------------------------

def load_data(csv_path, batch_size, train_usage='Training', val_usage='PublicTest'):
    """
    从带 Usage 列的 CSV 文件加载训练集和验证集
    """
    train_dataset = EmotionCSVDataset(csv_path, transform=train_transform, usage_filter=train_usage)
    val_dataset = EmotionCSVDataset(csv_path, transform=val_transform, usage_filter=val_usage)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
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
# ------------------------- 5. 训练进程控制 -------------------------
# 定义训练状态
class TrainState(Enum):
    RUNNING = 1
    PAUSED = 2
    STOPPED = 3

# 全局状态和同步机制
train_state = TrainState.RUNNING
state_lock = threading.Lock()
pause_condition = threading.Condition(state_lock)   # 用于暂停/继续时的等待与通知

# 三个控制函数
def pause_training():
    """暂停训练：将状态设为 PAUSED"""
    with state_lock:
        global train_state
        train_state = TrainState.PAUSED
        print("\n[控制] 训练已暂停。输入 '.' 继续，'/' 中断。")

def resume_training():
    """继续训练：将状态设为 RUNNING，并唤醒等待的线程"""
    with state_lock:
        global train_state
        if train_state == TrainState.PAUSED:
            train_state = TrainState.RUNNING
            pause_condition.notify_all()   # 唤醒所有在等待的循环
            print("\n[控制] 训练已继续。")

def stop_training():
    """中断训练：将状态设为 STOPPED，并唤醒等待的线程"""
    with state_lock:
        global train_state
        train_state = TrainState.STOPPED
        pause_condition.notify_all()
        print("\n[控制] 训练已中断。正在保存当前模型并退出...")

# 后台监听线程（接收控制台命令）
def control_listener():
    """在后台线程中运行，读取用户输入并调用控制函数"""
    while True:
        cmd = input().strip().lower()
        if cmd == ',':
            pause_training()
        elif cmd == '.':
            resume_training()
        elif cmd == '/':
            stop_training()
            break   # 中断后退出监听线程
        else:
            print("未知命令。可用命令: pause(,), resume(.), stop(/)")

# 训练循环中的状态检查辅助函数
def check_training_state():
    """
    在每次 batch 或 epoch 开始前调用。
    如果状态为 PAUSED，则阻塞直到恢复或被中断。
    如果状态为 STOPPED，则返回 False 表示应终止训练。
    若为 RUNNING，返回 True。
    """
    with state_lock:
        while train_state == TrainState.PAUSED:
            pause_condition.wait()          # 释放锁并等待通知
        if train_state == TrainState.STOPPED:
            return False
    return True


# ------------------------- 6. 训练与验证函数 -------------------------
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

# ------------------------- 7. 主训练流程 -------------------------
def train(csv_path):
    # 加载数据（请替换为实际数据路径）
    train_loader, val_loader = load_data(
        csv_path, 
        batch_size=BATCH_SIZE,
        train_usage='Training',
        val_usage='PublicTest'
    )
    
    # 实例化模型、损失函数、优化器
    model = EmotionCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()   # 多分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    # 启动后台控制监听线程
    listener_thread = threading.Thread(target=control_listener, daemon=True)
    listener_thread.start()
    print("训练控制台已启动。输入指令: pause(,), resume(.), stop(/)")
    # 记录训练过程
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    completed_epochs = 0
    for epoch in range(EPOCHS):
        completed_epochs = epoch + 1
        # 在每个 epoch 开始前检查是否要暂停/停止
        if not check_training_state():
            print("训练因用户中断而终止。")
            break

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
        
         # 每完成一个 epoch 后也检查一次（支持在 epoch 之间暂停）
        if not check_training_state():
            print("训练因用户中断而终止。")
            break

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "model_cnn.pth")
            print(f"  --> 保存最佳模型，验证准确率: {val_acc:.4f}")

    # 训练结束，保存最后模型（如果中断也可保存）
    torch.save(model.state_dict(), "last_model.pth")
    if completed_epochs == EPOCHS:
        print("训练结束。")
    else:
        print("训练中断。")

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
    class_names = val_loader.dataset.classes
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

# ------------------------- 8. 单张图片预测示例 -------------------------
def predict(model_path, image_path, device=DEVICE):
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
    # 训练
    train("fer2013.csv")
    
    # 预测（仅当模型文件存在且测试图片存在时）
    model_file = "model_cnn.pth"
    test_image = "test_face.jpg"
    
    if os.path.exists(model_file) and os.path.exists(test_image):
        pred, prob = predict(model_file, test_image)
        print(f"预测情绪: {pred}, 概率分布: {prob}")
    else:
        print("模型文件或测试图片不存在。")