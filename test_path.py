import os

LFW_IMG_DIR = r"C:\Users\Lenovo\Desktop\bs\lfw112\lfw_images"
PAIRS_PATH = r"C:\Users\Lenovo\Desktop\bs\lfw112\pairs.txt"

print("1. 检查图片文件夹是否存在:", os.path.exists(LFW_IMG_DIR))
print("2. 检查pairs文件是否存在:", os.path.exists(PAIRS_PATH))

if os.path.exists(LFW_IMG_DIR):
    folders = [f for f in os.listdir(LFW_IMG_DIR) if os.path.isdir(os.path.join(LFW_IMG_DIR, f))]
    print(f"3. 找到 {len(folders)} 个子文件夹")
    print(f"   前5个: {folders[:5]}")