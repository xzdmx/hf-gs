import os
import sys
from PIL import Image

# 检查用户是否提供了足够的参数
if len(sys.argv) != 4:
    print("Usage: python3 {} <pseudo_mask_folder> <out_dir> <key_frames_path>".format(sys.argv[0]))
    sys.exit(1)

# 获取命令行参数
in_dir = os.path.join(sys.argv[1], 'label')
out_dir = sys.argv[2]
key_frames_path = sys.argv[3]

# 检查并创建 lama_inpaint 目录
lama_inpaint_dir = os.path.join(out_dir, 'lama_inpaint')
os.makedirs(lama_inpaint_dir, exist_ok=True)

# 读取 key_frames.txt 文件中的名称
with open(key_frames_path, 'r') as f:
    key_frame_names = [line.strip() for line in f.readlines()]

# 获取输入目录中的文件列表
in_names = sorted(os.listdir(in_dir))

# 确保输入文件数量和 key_frames.txt 中的名称数量一致
assert len(in_names) == len(key_frame_names), "The number of files in input directory should match the number of names in key_frames.txt!"

# 遍历输入文件
for i, name in enumerate(in_names):
    src_path = os.path.join(in_dir, name)
    image = Image.open(src_path)

    # 构造目标文件名，使用 key_frames.txt 中的名称
    tgt_name = f"{key_frame_names[i]}.png"  # 假设保存为 PNG 格式，可按需修改
    tgt_path = os.path.join(lama_inpaint_dir, tgt_name)

    # 保存图像到目标路径
    image.save(tgt_path)
    print("Copy ", src_path, "........to........", tgt_path)