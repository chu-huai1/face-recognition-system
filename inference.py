import os
import cv2
import numpy as np
import onnxruntime as ort
import pickle
from pathlib import Path
from mtcnn import MTCNN
from preprocess.retinex import retinex_enhance   # 新增 Retinex 增强

# ====================== 核心配置 ======================
THRESHOLD = 0.55  # 识别阈值（可调整）
FEATURE_DB = "face_database.pkl"

# 模型路径（使用您下载的 MobileFaceNet 模型）
BASE_DIR = Path(__file__).parent
RECOGNIZER_MODEL = str(BASE_DIR / "weights" / "w600k_mbf.onnx")

class FaceRecognizer:
    def __init__(self):
        print("正在初始化 FaceID 系统...")
        
        # 加载 ONNX 识别模型
        if os.path.exists(RECOGNIZER_MODEL):
            self.recognizer = ort.InferenceSession(
                RECOGNIZER_MODEL,
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.recognizer.get_inputs()[0].name
            print(f"已加载识别模型: {Path(RECOGNIZER_MODEL).name}")
        else:
            raise FileNotFoundError(f"未找到识别模型: {RECOGNIZER_MODEL}")
        
        # 加载 MTCNN 检测器
        self.detector = MTCNN()
        print("已加载 MTCNN 检测器")
        
        # 加载人脸数据库
        self.face_db = self._load_db()
        self.last_feature = None
        print(f"系统启动成功！已注册人数: {len(self.face_db)}")

    def _load_db(self):
        """加载人脸特征数据库"""
        if os.path.exists(FEATURE_DB):
            try:
                with open(FEATURE_DB, "rb") as f:
                    return pickle.load(f)
            except:
                print("数据库文件损坏，将创建新数据库")
                return {}
        return {}

    def _save_db(self):
        """保存人脸特征数据库"""
        try:
            with open(FEATURE_DB, "wb") as f:
                pickle.dump(self.face_db, f)
        except:
            print("保存数据库失败")

    def _detect_face(self, img):
        """使用 MTCNN 检测人脸并返回对齐后的图像"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(img_rgb)
        
        if not results:
            return None, None
        
        best = max(results, key=lambda x: x['confidence'])
        x, y, w, h = best['box']
        keypoints = best['keypoints']
        
        x, y = max(0, x), max(0, y)
        
        landmarks = np.array([
            keypoints['left_eye'],
            keypoints['right_eye'],
            keypoints['nose'],
            keypoints['mouth_left'],
            keypoints['mouth_right']
        ], dtype=np.float32)
        
        aligned = self._align_face(img, landmarks)
        return (x, y, w, h), aligned

    def _align_face(self, img, landmarks):
        """使用5个关键点对齐人脸到 112x112"""
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        dst = landmarks.astype(np.float32)
        
        M, _ = cv2.estimateAffine2D(dst, src)
        if M is None:
            return cv2.resize(img, (112, 112))
        
        aligned = cv2.warpAffine(img, M, (112, 112))
        return aligned

    def _preprocess(self, face_img):
        """
        预处理：伽马变换 + Retinex 增强 + 归一化
        """
        face_img = cv2.resize(face_img, (112, 112))
        
        # 1. 伽马变换（光照归一化）
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        mean_gray = np.mean(gray)
        if mean_gray < 80:
            gamma = 0.6
        elif mean_gray > 180:
            gamma = 1.4
        else:
            gamma = 1.0
        look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
        face_img = cv2.LUT(face_img, look_up_table)
        
        # 2. Retinex 增强
        face_img = retinex_enhance(face_img, sigma=15)
        
        # 3. 转换为 RGB 并归一化到 [-1, 1]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.astype(np.float32) / 127.5 - 1.0
        face_img = face_img.transpose(2, 0, 1)
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def _get_feature(self, face_img):
        """提取特征向量并强制 L2 归一化"""
        blob = self._preprocess(face_img)
        feature = self.recognizer.run(None, {self.input_name: blob})[0]
        feature = feature.flatten()
        # 强制 L2 归一化
        feature = feature / (np.linalg.norm(feature) + 1e-8)
        self.last_feature = feature
        return feature

    def _compute_similarity(self, feat1, feat2):
        """计算余弦相似度"""
        return float(np.dot(feat1, feat2))

    def get_last_feature(self):
        """获取最后提取的特征"""
        return self.last_feature.tolist() if self.last_feature is not None else None

    def recognize(self, img_path):
        """1:N 人脸识别，增加 Top-3 验证机制"""
        if not self.face_db:
            return {"name": "未识别", "similarity": 0.0, "success": False, "error": "人脸库为空"}

        img = cv2.imread(img_path)
        if img is None:
            return {"name": "图片错误", "similarity": 0.0, "success": False, "error": "无法读取图片"}

        bbox, aligned = self._detect_face(img)
        if bbox is None:
            return {"name": "未检测到人脸", "similarity": 0.0, "success": False, "error": "未检测到人脸"}

        query_feat = self._get_feature(aligned)

        # 计算与库中所有人脸的相似度
        similarities = []
        for key, data in self.face_db.items():
            db_feat = data.get('feature')
            name = data.get('name', key)
            if db_feat is None:
                continue
            if isinstance(db_feat, list):
                db_feat = np.array(db_feat)
            sim = self._compute_similarity(query_feat, db_feat)
            similarities.append((key, name, sim))
        
        if not similarities:
            return {"name": "未识别", "similarity": 0.0, "success": False, "error": "人脸库无有效特征"}

        # 按相似度降序排序，取 Top-3
        similarities.sort(key=lambda x: x[2], reverse=True)
        top3 = similarities[:3]
        best_key, best_name, best_sim = top3[0]
        similarity_percent = round(best_sim * 100, 1)

        # Top-3 验证规则：最高分与第二高分差值 < 0.1 且最高分 < 0.65 时拒绝
        if len(top3) >= 2:
            second_sim = top3[1][2]
            if best_sim - second_sim < 0.1 and best_sim < 0.65:
                return {
                    "name": "未识别",
                    "similarity": similarity_percent,
                    "success": False,
                    "error": "结果置信度不足（Top-2 差异过小）"
                }

        # 阈值判断
        if best_sim >= THRESHOLD:
            return {
                "name": best_name,
                "similarity": similarity_percent,
                "success": True
            }
        else:
            return {
                "name": "未识别",
                "similarity": similarity_percent,
                "success": False,
                "error": f"未在人脸库中识别出该人脸（最高相似度 {similarity_percent}% < 阈值 {THRESHOLD*100}%）"
            }

    def verify(self, img_path1, img_path2):
        """1:1 人脸验证"""
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        
        if img1 is None or img2 is None:
            return 0.0
        
        _, aligned1 = self._detect_face(img1)
        _, aligned2 = self._detect_face(img2)
        
        if aligned1 is None or aligned2 is None:
            return 0.0
        
        feat1 = self._get_feature(aligned1)
        feat2 = self._get_feature(aligned2)
        
        sim = self._compute_similarity(feat1, feat2)
        return float(round(sim * 100, 1))

    def register(self, img_path, name, face_id=None):
        """注册新的人脸"""
        # 检查重复姓名
        for key, val in self.face_db.items():
            if val.get('name') == name:
                print(f"姓名 {name} 已存在")
                return False

        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            return False
        
        bbox, aligned = self._detect_face(img)
        if bbox is None:
            print(f"未检测到人脸")
            return False
        
        feature = self._get_feature(aligned)
        
        key = str(face_id) if face_id else name
        
        self.face_db[key] = {
            'name': name,
            'feature': feature,
            'id': face_id
        }
        
        self._save_db()
        print(f"成功注册: {name}")
        return True

    def delete_face(self, face_id):
        """删除人脸记录"""
        key = str(face_id)
        if key in self.face_db:
            del self.face_db[key]
            self._save_db()
            return True
        return False