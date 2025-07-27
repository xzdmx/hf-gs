import numpy as np
import cv2
import os


def save_depth_as_colorful_image(depth, output_path):
    """
    将深度图转换为彩色图像并保存

    参数:
        depth: numpy数组，深度图数据
        output_path: 输出图像路径
    """
    if depth.size == 0:
        raise ValueError("深度图为空")

    # 归一化深度值到[0,1]范围
    img = depth.astype(np.float32)
    img_min = img.min()
    img_max = img.max()

    if img_max == img_min:
        # 处理所有值都相同的情况
        img = np.zeros_like(img)
    else:
        img = (img - img_min) / (img_max - img_min)

    # 转换为8位无符号整数 [0, 255]
    img = (img * 255.0).astype(np.uint8)

    # 应用彩色映射 (JET颜色方案)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    # 保存图像
    cv2.imwrite(output_path, img)
    print(f"已保存彩色深度图到: {output_path}")


def main():
    # 直接在代码中指定输入输出路径
    input_path = '/media/junz/4TB-1/ldh/papers/0_Infusion/output/in2n-data/bear_1/depth_completed/depth_completed_frame_00041/frame_00041_depth.npy'
    output_path = '/media/junz/4TB-1/ldh/papers/0_Infusion/output/in2n-data/bear_1/depth_completed/depth_completed_frame_00041/frame_00041_colored.png'

    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件 '{input_path}' 不存在")

        # 读取深度图
        depth = np.load(input_path)

        # 保存为彩色图像
        save_depth_as_colorful_image(depth, output_path)

    except ValueError as ve:
        print(f"值错误: {ve}")
    except FileNotFoundError as fnf:
        print(f"文件错误: {fnf}")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    main()