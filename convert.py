import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

project_path = os.path.join(current_dir, 'weights', 'arcface-pytorch')
if project_path not in sys.path:
    sys.path.insert(0, project_path)
    print(f"✅ 已添加项目路径: {project_path}")

try:
    from nets.mobilefacenet import MobileFaceNet
    print("✅ 成功导入 MobileFaceNet 模型")
except ModuleNotFoundError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

print("正在加载权重文件...")

# ⭐ 关键修改：embedding_size=512
model = MobileFaceNet(embedding_size=512)

weights_path = os.path.join(current_dir, 'weights', 'arcface_mobilefacenet.pth')
if not os.path.exists(weights_path):
    print(f"❌ 找不到权重文件: {weights_path}")
    sys.exit(1)

# 加载权重
state_dict = torch.load(weights_path, map_location="cpu")

# 处理可能的多GPU训练时的 key 前缀 'module.'
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_k = k[7:]
    else:
        new_k = k
    new_state_dict[new_k] = v

# 加载权重，strict=False 允许不完全匹配
missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
if missing:
    print(f"⚠️ 缺失的键数量: {len(missing)}")
if unexpected:
    print(f"⚠️ 多余的键数量: {len(unexpected)}")

model.eval()
print("✅ 权重加载成功")

# 转换 ONNX
print("正在转换为 ONNX...")
dummy_input = torch.randn(1, 3, 112, 112)
onnx_path = os.path.join(current_dir, 'weights', 'mobilefacenet_fp32.onnx')
torch.onnx.export(
    model, dummy_input, onnx_path,
    input_names=["input"], output_names=["output"],
    opset_version=11, do_constant_folding=True
)
print(f"✅ 转换完成: {onnx_path}")