import onnxruntime as ort
import numpy as np

# 加载新模型
sess = ort.InferenceSession("weights/mobilefacenet_fp32.onnx", providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name

# 创建一个随机输入，模拟预处理后的图片
dummy_input = np.random.randn(1, 3, 112, 112).astype(np.float32)

# 提取特征
output = sess.run(None, {input_name: dummy_input})[0]
feature = output.flatten()

# 打印关键信息
print(f"特征形状: {feature.shape}")
print(f"特征范数 (L2 norm): {np.linalg.norm(feature):.6f}")
print(f"特征前10个值: {feature[:10]}")