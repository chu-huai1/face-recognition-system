import os
import cv2
import numpy as np
import onnxruntime as ort

LFW_BASE = "lfw112/lfw_images"
PAIRS_PATH = "lfw112/pairs.txt"
MODEL_PATH = "weights/w600k_mbf.onnx"

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
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    feat = session.run(None, {input_name: img})[0]
    feat = feat.flatten()
    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return feat

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

# 先找最佳阈值
print("\n计算最佳阈值...")
same_sims = []
diff_sims = []
for i in range(min(500, len(test_pairs))):
    img1_path, img2_path, label = test_pairs[i]
    f1, f2 = get_feature(img1_path), get_feature(img2_path)
    if f1 is None or f2 is None:
        continue
    sim = np.dot(f1, f2)
    if label == 1:
        same_sims.append(sim)
    else:
        diff_sims.append(sim)

print(f"同一人: {len(same_sims)}对, 平均={np.mean(same_sims):.3f}, 范围=[{np.min(same_sims):.3f}, {np.max(same_sims):.3f}]")
print(f"不同人: {len(diff_sims)}对, 平均={np.mean(diff_sims):.3f}, 范围=[{np.min(diff_sims):.3f}, {np.max(diff_sims):.3f}]")

best_acc = 0
best_th = 0.55
for th in np.arange(0.3, 0.8, 0.01):
    acc = (sum(1 for s in same_sims if s >= th) + sum(1 for s in diff_sims if s < th)) / (len(same_sims) + len(diff_sims))
    if acc > best_acc:
        best_acc = acc
        best_th = th

print(f"最佳阈值: {best_th:.2f}, 准确率: {best_acc*100:.2f}%")

input(f"\n按回车使用阈值 {best_th:.2f} 测试全部6000对...")

correct = 0
total = len(test_pairs)
for i, (img1_path, img2_path, label) in enumerate(test_pairs):
    if (i+1) % 500 == 0:
        print(f"进度: {i+1}/{total}")
    
    f1, f2 = get_feature(img1_path), get_feature(img2_path)
    if f1 is None or f2 is None:
        continue
    
    sim = np.dot(f1, f2)
    pred = 1 if sim >= best_th else 0
    if pred == label:
        correct += 1

accuracy = correct / total
print(f"\n========== 测试结果 ==========")
print(f"模型: {MODEL_PATH}")
print(f"LFW准确率: {accuracy * 100:.2f}%")