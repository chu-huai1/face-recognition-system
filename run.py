import os
import cv2
import numpy as np
import onnxruntime as ort

# ===== 您的路径 =====
LFW_IMG_DIR = r"C:\Users\Lenovo\Desktop\bs\lfw112\lfw_images"
PAIRS_PATH = r"C:\Users\Lenovo\Desktop\bs\lfw112\pairs.txt"
MODEL_PATH = r"C:\Users\Lenovo\Desktop\bs\weights\w600k_mbf.onnx"
# ===================

def preprocess(img):
    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, :] / 127.5 - 1
    return img

print("加载模型...")
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

print("加载pairs...")
pairs = []
with open(PAIRS_PATH, 'r') as f:
    lines = f.readlines()

# 第一行是说明，跳过
for line in lines[1:]:
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    if len(parts) == 3:
        # 同一个人：name, idx1, idx2
        name, idx1, idx2 = parts[0], int(parts[1]), int(parts[2])
        pairs.append((1, name, idx1, name, idx2))
    elif len(parts) == 4:
        # 不同人：name1, idx1, name2, idx2
        name1, idx1, name2, idx2 = parts[0], int(parts[1]), parts[2], int(parts[3])
        pairs.append((0, name1, idx1, name2, idx2))

print(f"共 {len(pairs)} 个测试对")

correct = 0
total = 0

for i, (label, name1, idx1, name2, idx2) in enumerate(pairs):
    # 构造图片路径
    img1_path = os.path.join(LFW_IMG_DIR, name1, f"{name1}_{idx1:04d}.jpg")
    img2_path = os.path.join(LFW_IMG_DIR, name2, f"{name2}_{idx2:04d}.jpg")
    
    # 检查文件是否存在
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        continue
    
    # 读取图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        continue
    
    # 提取特征
    feat1 = sess.run(None, {input_name: preprocess(img1)})[0].flatten()
    feat2 = sess.run(None, {input_name: preprocess(img2)})[0].flatten()
    
    # 计算余弦相似度
    sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8)
    pred = 1 if sim > 0.5 else 0
    
    if pred == label:
        correct += 1
    total += 1
    
    # 每500对打印一次进度
    if total % 500 == 0:
        print(f"进度: {total}/{len(pairs)}")

print(f"\n准确率: {correct/total*100:.2f}% ({correct}/{total})")