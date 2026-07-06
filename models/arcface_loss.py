import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5, sub_centers=1):
        """
        in_features: 输入特征维度（512）
        out_features: 类别数（训练时使用）
        s: 缩放因子
        m: 基础角度边际（动态时会覆盖）
        sub_centers: 每个类别的子中心数（Sub-Center ArcFace）
        """
        super(ArcFace, self).__init__()
        self.s = s
        self.base_m = m
        self.sub_centers = sub_centers
        self.out_features = out_features

        # 权重 shape: (out_features * sub_centers, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(out_features * sub_centers, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels, current_epoch=None, total_epochs=50):
        """
        x: (batch, 512) 已 L2 归一化的特征
        labels: (batch,) 类别索引
        current_epoch, total_epochs: 用于动态角度边际
        """
        # 动态边际：初期 0.3，后期 0.5
        if current_epoch is not None and total_epochs is not None:
            m = min(0.5, 0.3 + 0.2 * (current_epoch / total_epochs))
        else:
            m = self.base_m

        # L2 归一化权重
        w = F.normalize(self.weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        cos_theta = torch.matmul(x, w.t())  # (batch, out_features * sub_centers)

        # Sub-Center 处理：取每个类别中最大 cos 值
        if self.sub_centers > 1:
            cos_theta = cos_theta.view(-1, self.out_features, self.sub_centers)
            cos_theta, _ = torch.max(cos_theta, dim=2)
            cos_theta = cos_theta.view(-1, self.out_features)

        # 计算角度并增加边际
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + m)
        logits = cos_theta * 1.0
        logits.scatter_(1, labels.view(-1, 1), target_logits)
        logits *= self.s

        # L2 正则化损失（系数 0.001，与交叉熵损失相加）
        reg_loss = 0.001 * torch.norm(self.weight, p=2)

        return logits, reg_loss