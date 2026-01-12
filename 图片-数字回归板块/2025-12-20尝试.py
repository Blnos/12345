import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm  # 进度条显示

# ===================== 1. 配置参数（根据实际情况修改）=====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先用GPU
DATA_ROOT = "./data"  # 数据根目录（对应上面的data文件夹）
IMAGE_DIR = os.path.join(DATA_ROOT, "images")  # 图片文件夹路径
LABEL_PATH = os.path.join(DATA_ROOT, "labels.csv")  # 标签文件路径

# 训练参数
BATCH_SIZE = 8  # 批次大小（GPU内存小就改4/2）
EPOCHS = 50  # 最大训练轮数
LEARNING_RATE = 1e-4  # 学习率
PATIENCE = 8  # 早停耐心值（连续8轮验证集损失不下降则停止）
IMAGE_SIZE = (480, 320)  # 直接用原始尺寸（宽480，高320），无需缩放



# ===================== 2. 自定义数据集类（加载图片和标签）=====================
class StreetComplaintDataset(Dataset):
    def __init__(self, image_names, complaint_counts, transform=None):
        self.image_names = image_names  # 图片文件名列表
        self.complaint_counts = complaint_counts  # 对应投诉数列表
        self.transform = transform  # 图片预处理变换

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 加载图片
        img_path = os.path.join(IMAGE_DIR, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")  # 强制转为RGB（避免灰度图报错）

        # 加载标签（投诉数，转为float32用于回归）
        label = torch.tensor(self.complaint_counts[idx], dtype=torch.float32)

        # 图片预处理
        if self.transform:
            image = self.transform(image)

        return image, label


# ===================== 3. 数据加载与预处理 =====================
def load_data():
    # 读取标签文件
    df = pd.read_csv(LABEL_PATH)
    image_names = df["image_name"].values  # 所有图片名
    complaint_counts = df["complaint_count"].values  # 所有投诉数

    # 划分训练集（80%）和验证集（20%）
    train_names, val_names, train_labels, val_labels = train_test_split(
        image_names, complaint_counts, test_size=0.2, random_state=42  # 随机种子保证可复现
    )

    # 图片预处理：训练集用数据增强，验证集仅归一化
    # 训练集预处理（数据增强+保持3:2比例）
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),  # 若目标尺寸是480×320，原始图已是这个尺寸，相当于无操作
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转（不改变比例）
        transforms.RandomRotation(degrees=10),  # 随机旋转（不改变比例）
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 验证集预处理（同理）
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集实例
    train_dataset = StreetComplaintDataset(train_names, train_labels, train_transform)
    val_dataset = StreetComplaintDataset(val_names, val_labels, val_transform)

    # 创建数据加载器（批量加载+打乱）
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, df["complaint_count"].max()  # 返回最大投诉数（用于后续结果可视化）


# ===================== 4. 定义回归模型（ResNet50+回归头）=====================
def build_model():
    model = models.resnet50(pretrained=True)
    # 冻结底层卷积层（仅训练顶层回归头）
    for param in model.parameters():
        param.requires_grad = False

    # 关键修改：计算非正方形输入的全连接层输入维度
    input_width = IMAGE_SIZE[0]  # 480
    input_height = IMAGE_SIZE[1]  # 320
    final_width = input_width // 32  # 480÷32=15
    final_height = input_height // 32  # 320÷32=10
    num_ftrs = model.fc.in_features  # ResNet50固定通道数=2048
    new_fc_input = num_ftrs * final_width * final_height  # 2048×15×10=307200

    # 替换回归头（输入维度改为new_fc_input）
    model.fc = nn.Sequential(
        nn.Linear(new_fc_input, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

    model = model.to(DEVICE)
    return model


# ===================== 5. 训练函数（含早停）=====================
def train_model(model, train_loader, val_loader):
    # 定义损失函数（MSE：均方误差，适配回归）
    criterion = nn.MSELoss()
    # 定义优化器（Adam：自适应学习率，效果好）
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)  # 仅优化顶层回归头

    # 记录训练过程（用于后续可视化）
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")  # 最佳验证损失（初始设为无穷大）
    early_stop_counter = 0  # 早停计数器

    # 开始训练
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()  # 设为训练模式（启用dropout）
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} (Train)"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)  # 标签维度从[N]转为[N,1]（匹配模型输出）

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播+优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播求梯度
            optimizer.step()  # 更新参数

            # 累计训练损失
            train_loss += loss.item() * images.size(0)

        # 计算训练集平均损失
        train_loss_avg = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss_avg)

        # 验证阶段
        model.eval()  # 设为验证模式（禁用dropout）
        val_loss = 0.0
        with torch.no_grad():  # 禁用梯度计算（节省内存+加速）
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} (Val)"):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        # 计算验证集平均损失
        val_loss_avg = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss_avg)

        # 打印当前轮次结果
        print(f"Epoch {epoch + 1} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")

        # 早停逻辑：如果验证损失下降，保存最佳模型；否则计数
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), "best_complaint_model.pth")  # 保存最佳模型权重
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {PATIENCE} epochs)")
                break

    # 绘制训练/验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()

    return model


# ===================== 6. 模型评估（计算回归指标）=====================
def evaluate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images).squeeze(1)  # 输出维度从[N,1]转为[N]

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算回归指标
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)  # 均方根误差（更易理解，单位与投诉数一致）
    r2 = r2_score(all_labels, all_preds)  # 决定系数（0-1，越接近1越好）

    print("\n模型评估结果：")
    print(f"RMSE (均方根误差): {rmse:.2f}")
    print(f"R² (决定系数): {r2:.4f}")

    # 绘制真实值vs预测值散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(all_labels, all_preds, alpha=0.6)
    plt.plot([0, max(all_labels)], [0, max(all_labels)], "r--")  # 理想预测线（y=x）
    plt.xlabel("真实投诉数")
    plt.ylabel("预测投诉数")
    plt.title("真实值 vs 预测值")
    plt.grid(True, alpha=0.3)
    plt.savefig("pred_vs_true.png")
    plt.show()


# ===================== 7. 预测函数（输入新图片，输出投诉数预估）=====================
def predict_complaints(model_path, image_path):
    # 加载模型
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))  # 加载最佳权重
    model.eval()  # 设为验证模式

    # 图片预处理（与验证集一致）
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载并预处理新图片
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # 增加batch维度（模型要求输入为[N,C,H,W]）
    image = image.to(DEVICE)

    # 预测
    with torch.no_grad():
        pred = model(image).item()  # 转为Python数值

    # 投诉数不能为负，修正为0
    pred_complaints = max(0, round(pred))  # 四舍五入为整数（投诉数是整数）
    print(f"\n输入图片的预估投诉数：{pred_complaints} 条")
    return pred_complaints


# ===================== 主函数（串联所有流程）=====================
if __name__ == "__main__":
    # 1. 加载数据
    train_loader, val_loader, max_complaint = load_data()
    print(f"数据加载完成！训练集样本数：{len(train_loader.dataset)}，验证集样本数：{len(val_loader.dataset)}")

    # 2. 构建模型
    model = build_model()
    print("模型构建完成！")

    # 3. 训练模型
    model = train_model(model, train_loader, val_loader)

    # 4. 评估模型
    evaluate_model(model, val_loader)

    # 5. 预测新图片（替换为你的新图片路径）
    new_image_path = "./test_street.jpg"  # 测试图片路径
    if os.path.exists(new_image_path):
        predict_complaints("best_complaint_model.pth", new_image_path)
    else:
        print(f"\n测试图片 {new_image_path} 不存在，跳过预测演示")
