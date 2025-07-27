import cv2
import numpy as np
from scipy import ndimage


def load_and_prepare_images(image_path, mask_path):
    """
    加载原始图像和mask
    """
    # 读取图像
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 确保mask是二值图像
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return image, mask


def enhance_texture(image, mask):
    """
    增强图像纹理
    """
    # 转换到LAB颜色空间以分离亮度通道
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # 在mask区域应用自适应直方图均衡
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = l_channel.copy()
    enhanced_l[mask > 0] = clahe.apply(l_channel)[mask > 0]

    # 应用锐化
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    enhanced_l = cv2.filter2D(enhanced_l, -1, kernel)

    # 更新LAB图像的L通道
    lab[:, :, 0] = enhanced_l

    # 转换回BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced


def detail_enhancement(image, mask):
    """
    细节增强
    """
    # 分离通道
    b, g, r = cv2.split(image)

    # 对每个通道进行处理
    for channel in [b, g, r]:
        # 在mask区域内应用细节增强
        mask_region = channel[mask > 0]
        if len(mask_region) > 0:
            # 使用双边滤波保持边缘
            detail = cv2.bilateralFilter(channel, 9, 75, 75)
            channel[mask > 0] = detail[mask > 0]

    # 合并通道
    enhanced = cv2.merge([b, g, r])

    return enhanced


def texture_synthesis(image, mask):
    """
    纹理合成和修复
    """
    # 创建修复蒙版
    repair_mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))

    # 使用Inpainting算法修复
    enhanced = cv2.inpaint(image, repair_mask, 3, cv2.INPAINT_TELEA)

    return enhanced


def main(image_path, mask_path, output_path):
    """
    主函数：结合所有增强步骤
    """
    # 加载图像和mask
    image, mask = load_and_prepare_images(image_path, mask_path)

    # 应用纹理增强
    enhanced = enhance_texture(image.copy(), mask)

    # 应用细节增强
    enhanced = detail_enhancement(enhanced, mask)

    # # 应用纹理合成
    # enhanced = texture_synthesis(enhanced, mask)

    # 混合原始图像和增强结果
    alpha = 0.3
    final_result = cv2.addWeighted(image, 1 - alpha, enhanced, alpha, 0)

    # 只在mask区域应用增强效果
    final_result[mask == 0] = image[mask == 0]

    # 保存结果
    cv2.imwrite(output_path, final_result)

    return final_result


# 使用示例
if __name__ == "__main__":
    image_path = "../output/in2n-data/bear/train/ours_150/renders/frame_00041.png"
    mask_path = "../output/in2n-data/bear/mask/sub_mask/frame_00041_small.png"
    output_path = "../output/in2n-data/bear/enhance_images/enhance_image.png"

    enhanced_image = main(image_path, mask_path, output_path)
