import argparse

import cv2
import numpy as np
# def combine_with_mask(image1, depth1, image2, depth2, mask):
#     combined_image = image1.copy()
#     combined_depth = depth1.copy()
#     h, w = image1.shape[:2]
#     for i in range(h):
#         for j in range(w):
#             if mask[i, j]:
#                 combined_image[i, j] = image2[i, j]
#                 combined_depth[i, j] = depth2[i, j]
#     return combined_image, combined_depth

def combine_with_mask(image1, image2, mask):
    combined_image = image1.copy()
    h, w = image1.shape[:2]
    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                combined_image[i, j] = image2[i, j]
    return combined_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image1_path', required=True, help='Path to the first input image.')
    # parser.add_argument('--input_depth1_path', required=True, help='Path to the first input depth data.')
    parser.add_argument('--input_image2_path', required=True, help='Path to the second input image.')
    # parser.add_argument('--input_depth2_path', required=True, help='Path to the second input depth data.')
    parser.add_argument('--input_mask_path', required=True, help='Path to the mask image.')
    parser.add_argument('--save_path', required=True, help='Path to the save image.')
    args = parser.parse_args()

    # 读取图像和深度数据
    image1 = cv2.imread(args.input_image1_path)
    image2 = cv2.imread(args.input_image2_path)

    # 读取掩码
    mask = cv2.imread(args.input_mask_path, cv2.IMREAD_GRAYSCALE)
    mask[mask != 0] = 1
    combined_image = combine_with_mask(image1, image2, mask)

    # 保存合并后的图像和深度到 input_image2_path 和 input_depth2_path
    cv2.imwrite(args.save_path, combined_image)