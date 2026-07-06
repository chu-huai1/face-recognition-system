import os
import cv2
import numpy as np
import onnxruntime as ort

LFW_IMG_DIR = r"C:\Users\Lenovo\Desktop\bs\lfw112\lfw_images"
PAIRS_PATH = r"C:\Users\Lenovo\Desktop\bs\lfw112\pairs.txt"
MODEL_PATH = r"C:\Users\Lenovo\Desktop\bs\weights\w600k_mbf.onnx"

def preprocess(img):
    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, :] / 127.5 - 1
    return img

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

pairs = []
with open(PAIRS_PATH, 'r') as f:
    for line in f.readlines()[1:]:
        parts = line.strip().split()
        if len(parts) == 3:
            pairs.append((1, parts[0], int(parts[1]), parts[0], int(parts[2])))
        else:
            pairs.append((0, parts[0], int(parts[1]), parts[2], int(parts[3])))

print("提取所有特征...")
sims = []
labels = []
for label, n1, i1, n2, i2 in pairs:
    p1 = os.path.join(LFW_IMG_DIR, n1, f"{n1}_{i1:04d}.jpg")
    p2 = os.path.join(LFW_IMG_DIR, n2, f"{n2}_{i2:04d}.jpg")
    if not os.path.exists(p1) or not os.path.exists(p2):
        continue
    img1, img2 = cv2.imread(p1), cv2.imread(p2)
    if img1 is None or img2 is None:
        continue
    f1 = sess.run(None, {input_name: preprocess(img1)})[0].flatten()
    f2 = sess.run(None, {input_name: preprocess(img2)})[0].flatten()
    sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
    sims.append(sim)
    labels.append(label)

print("搜索最优阈值...")
best_acc = 0
best_thresh = 0
for thresh in np.arange(0.3, 0.9, 0.01):
    pred = [1 if s > thresh else 0 for s in sims]
    acc = np.mean(np.array(pred) == np.array(labels))
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print(f"\n最优阈值: {best_thresh:.2f}")
print(f"最高准确率: {best_acc*100:.2f}%")