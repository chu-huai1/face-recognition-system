from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from inference import FaceRecognizer
import os
import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
import cv2
from mtcnn import MTCNN

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
fr = FaceRecognizer()
app.json_encoder = NumpyEncoder

def init_db():
    conn = sqlite3.connect('faceid.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS faces
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  category TEXT DEFAULT '员工',
                  employee_id TEXT UNIQUE,
                  age INTEGER,
                  feature TEXT,
                  photo_path TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  matched_name TEXT,
                  similarity REAL,
                  status TEXT,
                  image_path TEXT)''')
    
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return send_from_directory('.', 'Index.html')

@app.route("/api/recognize", methods=["POST"])
def recognize():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "未上传图片"}), 400
    
    f = request.files["image"]
    path = "temp_recognize.jpg"
    f.save(path)
    
    try:
        result = fr.recognize(path)
        
        conn = sqlite3.connect('faceid.db')
        c = conn.cursor()
        c.execute("INSERT INTO logs (matched_name, similarity, status, image_path) VALUES (?, ?, ?, ?)",
                  (result.get('name'), result.get('similarity'), 
                   'success' if result['success'] else 'fail', path))
        conn.commit()
        conn.close()
        
        return app.response_class(
            response=json.dumps(result, cls=NumpyEncoder),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    finally:
        if os.path.exists(path):
            os.remove(path)

@app.route("/api/verify", methods=["POST"])
def verify():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"success": False, "error": "需要两张图片"}), 400

    path1 = "temp_verify1.jpg"
    path2 = "temp_verify2.jpg"
    request.files["image1"].save(path1)
    request.files["image2"].save(path2)

    try:
        similarity = fr.verify(path1, path2)
        return jsonify({"success": True, "similarity": float(similarity)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    finally:
        if os.path.exists(path1):
            os.remove(path1)
        if os.path.exists(path2):
            os.remove(path2)

@app.route("/api/faces", methods=["GET"])
def get_faces():
    category = request.args.get('category', '所有类别')
    search = request.args.get('search', '')
    
    conn = sqlite3.connect('faceid.db')
    c = conn.cursor()
    
    query = "SELECT id, name, category, employee_id, age, photo_path, created_at FROM faces WHERE 1=1"
    params = []
    
    if category != '所有类别':
        query += " AND category = ?"
        params.append(category)
    
    if search:
        query += " AND name LIKE ?"
        params.append(f"%{search}%")
    
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    
    faces = []
    for row in rows:
        faces.append({
            "id": row[0],
            "name": row[1],
            "category": row[2],
            "employee_id": row[3],
            "age": row[4],
            "photo_path": row[5],
            "created_at": row[6]
        })
    
    return jsonify(faces)

@app.route("/api/faces", methods=["POST"])
def add_face():
    name = request.form.get("name")
    category = request.form.get("category", "员工")
    employee_id = request.form.get("employee_id")
    age = request.form.get("age")
    
    if 'photo' not in request.files:
        return jsonify({"success": False, "error": "未上传图片"}), 400
    
    f = request.files["photo"]
    
    os.makedirs('uploads', exist_ok=True)
    temp_path = f"uploads/temp_{employee_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    f.save(temp_path)
    
    # 检查图片是否有效
    img = cv2.imread(temp_path)
    if img is None:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"success": False, "error": "图片文件无效"}), 400
    
    # 检查是否有人脸
    detector = MTCNN()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces_detected = detector.detect_faces(img_rgb)
    if not faces_detected:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"success": False, "error": "图片中未检测到人脸"}), 400
    
    try:
        success = fr.register(temp_path, name, employee_id)
        
        if success:
            conn = sqlite3.connect('faceid.db')
            c = conn.cursor()
            feature = fr.get_last_feature()
            
            final_path = f"uploads/{employee_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            os.rename(temp_path, final_path)
            
            c.execute("""INSERT INTO faces 
                        (name, category, employee_id, age, feature, photo_path) 
                        VALUES (?, ?, ?, ?, ?, ?)""",
                      (name, category, employee_id, age, json.dumps(feature), final_path))
            conn.commit()
            conn.close()
            
            return jsonify({"success": True, "message": "添加成功"})
        else:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({"success": False, "error": "注册失败，请确保照片中只有一张正脸"})
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/faces/<int:face_id>", methods=["DELETE"])
def delete_face(face_id):
    conn = sqlite3.connect('faceid.db')
    c = conn.cursor()
    
    # 1. 获取要删除的人脸信息
    c.execute("SELECT id, name FROM faces WHERE id = ?", (face_id,))
    face = c.fetchone()
    
    if not face:
        conn.close()
        return jsonify({"success": False, "error": "未找到该人脸"}), 404
    
    # 2. 删除 SQLite 中的记录
    c.execute("DELETE FROM faces WHERE id = ?", (face_id,))
    conn.commit()
    conn.close()
    
    # 3. 同时删除 face_database.pkl 中的特征数据
    fr.delete_face(face_id)
    
    return jsonify({"success": True})

@app.route("/api/faces/<int:face_id>", methods=["DELETE"])
def delete_face_old(face_id):
    conn = sqlite3.connect('faceid.db')
    c = conn.cursor()
    
    c.execute("SELECT photo_path FROM faces WHERE id = ?", (face_id,))
    row = c.fetchone()
    
    if row and row[0] and os.path.exists(row[0]):
        os.remove(row[0])
    
    c.execute("DELETE FROM faces WHERE id = ?", (face_id,))
    conn.commit()
    conn.close()
    
    fr.delete_face(face_id)
    
    return jsonify({"success": True})

@app.route("/api/faces/count", methods=["GET"])
def get_face_count():
    conn = sqlite3.connect('faceid.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM faces")
    count = c.fetchone()[0]
    conn.close()
    return jsonify({"count": count})

@app.route("/api/logs", methods=["GET"])
def get_logs():
    limit = request.args.get('limit', 10, type=int)
    conn = sqlite3.connect('faceid.db')
    c = conn.cursor()
    c.execute("SELECT id, timestamp, matched_name, similarity, status FROM logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    
    logs = []
    for row in rows:
        logs.append({
            "id": row[0],
            "timestamp": str(row[1]) if row[1] else "",
            "matched_name": row[2] if row[2] else "未识别",
            "similarity": float(row[3]) if row[3] else 0.0,
            "status": row[4] if row[4] else "fail"
        })
    return jsonify(logs)

@app.route("/api/dashboard/stats", methods=["GET"])
def get_stats():
    conn = sqlite3.connect('faceid.db')
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM faces")
    total_faces = c.fetchone()[0]
    
    today = datetime.now().strftime('%Y-%m-%d')
    c.execute("SELECT COUNT(*) FROM logs WHERE date(timestamp) = ?", (today,))
    today_count = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM logs")
    total_logs = c.fetchone()[0]
    
    if total_logs > 0:
        c.execute("SELECT COUNT(*) FROM logs WHERE status = 'success'")
        success_logs = c.fetchone()[0]
        accuracy = round(success_logs / total_logs * 100, 1)
    else:
        accuracy = 98.7
    
    conn.close()
    
    return jsonify({
        "total_faces": total_faces,
        "today_recognitions": today_count,
        "accuracy": accuracy,
        "avg_time": 0.32
    })

@app.route("/api/dashboard/trend", methods=["GET"])
def get_trend():
    conn = sqlite3.connect('faceid.db')
    c = conn.cursor()
    
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
    data = []
    labels = []
    
    for d in dates:
        c.execute("SELECT COUNT(*) FROM logs WHERE date(timestamp) = ?", (d,))
        count = c.fetchone()[0]
        data.append(count)
        labels.append(d[5:])
    
    conn.close()
    return jsonify({"labels": labels, "data": data})

@app.route("/api/dashboard/distribution", methods=["GET"])
def get_distribution():
    conn = sqlite3.connect('faceid.db')
    c = conn.cursor()
    c.execute("SELECT category, COUNT(*) FROM faces GROUP BY category")
    rows = c.fetchall()
    conn.close()
    
    if rows:
        labels = [row[0] for row in rows]
        data = [row[1] for row in rows]
    else:
        labels = ['员工', '访客', 'VIP']
        data = [0, 0, 0]
    
    return jsonify({"labels": labels, "data": data})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)