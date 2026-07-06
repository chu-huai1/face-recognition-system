import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import MobileFaceNet  # 假设您已定义模型结构

# -------------------- 教师模型（已训练好的 FP32 模型）--------------------
teacher_model = MobileFaceNet(num_features=512).to(device)
teacher_model.load_state_dict(torch.load("checkpoint_teacher.pth", map_location=device))
teacher_model.eval()  # 冻结教师模型参数

# -------------------- 学生模型（轻量化，通道数减半）--------------------
student_model = MobileFaceNet(num_features=512, width_mult=0.5).to(device)  # 示例：通道缩减为一半

# -------------------- 蒸馏训练参数 --------------------
temperature = 4.0          # 蒸馏温度
alpha = 0.7                # 软标签损失权重
beta = 0.3                 # 硬标签（交叉熵）损失权重
learning_rate = 0.001
epochs = 30

optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
criterion_hard = nn.CrossEntropyLoss()   # 硬标签损失

# -------------------- 蒸馏损失函数（KL 散度）--------------------
def distillation_loss(student_logits, teacher_logits, labels, temp, alpha, beta):
    # 软标签损失：学生与教师输出的 KL 散度
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / temp, dim=1),
        F.softmax(teacher_logits / temp, dim=1)
    ) * (temp * temp)
    # 硬标签损失
    hard_loss = criterion_hard(student_logits, labels)
    return alpha * soft_loss + beta * hard_loss

# -------------------- 训练循环 --------------------
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(epochs):
    student_model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # 教师模型前向（不计算梯度）
        with torch.no_grad():
            teacher_logits = teacher_model(images)
        # 学生模型前向
        student_logits = student_model(images)
        # 计算蒸馏损失
        loss = distillation_loss(student_logits, teacher_logits, labels,
                                 temperature, alpha, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# 保存学生模型
torch.save(student_model.state_dict(), "student_distilled.pth")