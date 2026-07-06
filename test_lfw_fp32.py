import os
import cv2
import numpy as np
import onnxruntime as ort

# ========== 配置 ==========
LFW_BASE = "lfw112/lfw_images"
PAIRS_PATH = "lfw112/pairs.txt"
MODEL_PATH = "weights/mobilefacenet_fp32.onnx"
THRESHOLD = 0.55

print(f"加载模型: {MODEL_PATH}")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

def get_feature(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 127.5 - 1.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # 添加batch维度
    
    feature = session.run(None, {input_name: img})[0]
    # 输出形状是 (1, 512, 1, 1)，需要压平
    feature = feature.reshape(-1)  # 变成 (512,)
    
    # L2归一化
    norm = np.linalg.norm(feature) + 1e-8
    feature = feature / norm
    return feature

print("读取测试对...")
test_pairs = []
with open(PAIRS_PATH, 'r') as f:
    lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            name, idx1, idx2 = parts[0], int(parts[1]), int(parts[2])
            img1 = f"{LFW_BASE}/{name}/{name}_{idx1:04d}.jpg"
            img2 = f"{LFW_BASE}/{name}/{name}_{idx2:04d}.jpg"
            test_pairs.append((img1, img2, 1))
        elif len(parts) == 4:
            name1, idx1, name2, idx2 = parts[0], int(parts[1]), parts[2], int(parts[3])
            img1 = f"{LFW_BASE}/{name1}/{name1}_{idx1:04d}.jpg"
            img2 = f"{LFW_BASE}/{name2}/{name2}_{idx2:04d}.jpg"
            test_pairs.append((img1, img2, 0))

print(f"共 {len(test_pairs)} 对测试样本")

# 先测试前10对，看看特征是否有效
print("\n测试前10对，检查特征...")
for i in range(min(10, len(test_pairs))):
    img1_path, img2_path, label = test_pairs[i]
    feat1 = get_feature(img1_path)
    feat2 = get_feature(img2_path)
    if feat1 is not None and feat2 is not None:
        sim = np.dot(feat1, feat2)
        print(f"  第{i+1}对: 相似度={sim:.4f}, 标签={label}")
    else:
        print(f"  第{i+1}对: 读取失败")

input("\n特征看起来正常吗？按回车继续测试全部...")

# 全部测试
correct = 0
total = len(test_pairs)

for i, (img1_path, img2_path, label) in enumerate(test_pairs):
    if (i+1) % 500 == 0:
        print(f"进度: {i+1}/{total}")
    
    feat1 = get_feature(img1_path)
    feat2 = get_feature(img2_path)
    if feat1 is None or feat2 is None:
        continue
    
    sim = np.dot(feat1, feat2)
    pred = 1 if sim >= THRESHOLD else 0
    if pred == label:
        correct += 1

accuracy = correct / total
print(f"\n========== 测试结果 ==========")
print(f"LFW准确率: {accuracy * 100:.2f}%")