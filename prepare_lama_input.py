import os
import shutil
import sys

# 检查用户是否提供了足够的参数
if len(sys.argv) != 5:
    print("Usage: python3 {} <img_path> <mask_path> <lama_path> <key_frames_path>".format(sys.argv[0]))
    sys.exit(1)

# 获取命令行参数
image_dir = sys.argv[1]
mask_dir = sys.argv[2]
out_dir = sys.argv[3]
key_frame_path = sys.argv[4]
out_mask_dir = os.path.join(out_dir, "label")

# 确保目标目录和标签目录存在
os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

# 读取 key_frame.txt 文件中的图片名称
with open(key_frame_path, 'r') as f:
    key_frame_names = [line.strip() for line in f.readlines()]

# 遍历 key_frame_names 中的每个名称
for name in key_frame_names:
    # 构造图片文件名
    image_filename = f"{name}.png"
    image_src_path = os.path.join(image_dir, image_filename)
    if os.path.exists(image_src_path):
        image_dst_path = os.path.join(out_dir, image_filename)
        shutil.copy2(image_src_path, image_dst_path)
        print(f"Copied image: {image_src_path} -> {image_dst_path}")

    # 构造掩码文件名
    mask_filename = f"{name}.png"
    mask_src_path = os.path.join(mask_dir, mask_filename)
    if os.path.exists(mask_src_path):
        mask_dst_path = os.path.join(out_mask_dir, mask_filename)
        shutil.copy2(mask_src_path, mask_dst_path)
        print(f"Copied mask: {mask_src_path} -> {mask_dst_path}")