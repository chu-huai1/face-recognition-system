# 基于 MobileFaceNet 与 ArcFace 的人脸识别系统

毕业设计项目，实现轻量化高精度人脸识别。

## 技术栈
- Python 3.8+ / PyTorch / ONNX Runtime
- MobileFaceNet + ArcFace（特征提取）
- MTCNN（人脸检测）
- Flask（后端服务）
- SQLite（数据存储）

## 主要功能
- 人脸注册（照片录入 + 姓名绑定）
- 1:1 人脸比对（验证是否为同一人）
- 1:N 人脸识别（从库中检索身份）
- 识别记录查询

## 测试结果
- LFW 准确率：**98.98%**
- 模型大小：**4.2MB**（INT8量化后）
- CPU推理速度：**7.8ms/帧**
- 端到端响应：**≤168ms**

## 运行方式
1. 安装依赖：`pip install -r requirements.txt`
2. 下载预训练权重放入 `weights/` 目录（链接见下方）
3. 启动服务：`python app.py`
4. 浏览器打开：`http://localhost:5000`

## 模型权重下载
由于文件较大，模型权重未包含在代码仓库中，已上传至百度网盘：

链接：https://pan.baidu.com/s/1Bmnhem6X16P57SUaWFEJ5w?pwd=ifs7
提取码：ifs7

下载后将 `mobilefacenet_fp32.onnx` 放入项目根目录下的 `weights/` 文件夹中即可运行。
