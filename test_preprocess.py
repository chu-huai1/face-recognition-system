import cv2
import numpy as np
from mtcnn import MTCNN
from inference import FaceRecognizer

# 初始化
detector = MTCNN()
fr = FaceRecognizer()

# 读一张有人脸的照片（改成你的图片路径）
img_path = r"C:\Users\Lenovo\Desktop\face.png"
img = cv2.imread(img_path)

if img is None:
    print("图片读取失败，请检查路径")
    exit()

# 1. 原图保存
cv2.imwrite("output_1_original.jpg", img)
print("已保存: output_1_original.jpg")

# 2. 检测人脸并裁剪
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = detector.detect_faces(img_rgb)

if not results:
    print("未检测到人脸")
    exit()

best = max(results, key=lambda x: x['confidence'])
x, y, w, h = best['box']
x, y = max(0, x), max(0, y)
face_crop = img[y:y+h, x:x+w]
cv2.imwrite("output_2_face_crop.jpg", face_crop)
print("已保存: output_2_face_crop.jpg")

# 3. 姿态矫正
keypoints = best['keypoints']
landmarks = np.array([
    keypoints['left_eye'],
    keypoints['right_eye'],
    keypoints['nose'],
    keypoints['mouth_left'],
    keypoints['mouth_right']
], dtype=np.float32)
aligned = fr._align_face(img, landmarks)
cv2.imwrite("output_3_aligned.jpg", aligned)
print("已保存: output_3_aligned.jpg")

# 4. Retinex增强
enhanced = fr._preprocess(aligned)
# _preprocess 返回的是归一化后的张量，需要转回图片
enhanced_img = ((enhanced[0].transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("output_4_retinex.jpg", enhanced_img)
print("已保存: output_4_retinex.jpg")

print("预处理完成！共生成4张图片")