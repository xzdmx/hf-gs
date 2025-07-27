import cv2
import numpy as np
import os
import argparse

def load_mask(mask_path):
    """读取mask，并转换为0，1mask"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask[mask != 0] = 1  # 将掩码所有非0处转换为1
    return mask


def mask_edge_expend(mask, size):
    """扩大mask的边界，size为扩大的尺寸,即size个像素"""
    kernel = np.ones((size, size), np.uint8)
    mask_expend = cv2.dilate(mask, kernel, iterations=1)
    return mask_expend

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Path to the input file or directory.')
    parser.add_argument('--output_dir', required=True, help='Path to the output file or directory.')
    parser.add_argument('--size', type=int, default=40, help='Size for expansion. Can be adjusted as needed.')
    args = parser.parse_args()

    # 在这里使用 args.input_path 和 args.output_path 进行后续操作
    print(f'----------------------------expand mask operation---------------------------------')
    print(f'Input path: {args.input_dir}')
    print(f'Output path: {args.output_dir}')
    # 如果输出目录不存在则创建目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 将seg目录下的mask（初始分割的mask）全部扩大，seg目录下mask格式为frame_*****.png
    seg_dir = args.input_dir
    save_dir = args.output_dir
    size = args.size  # 可以根据需要调整扩大的尺寸

    # 遍历seg目录下的所有mask图片
    for filename in os.listdir(seg_dir):
        if filename.endswith('.png'):
            mask_path = os.path.join(seg_dir, filename)
            mask = load_mask(mask_path)
            mask_expend = mask_edge_expend(mask, size).astype(np.uint8) * 255

            # 保存结果到save_path，文件名为原文件名
            save_filename = f"{os.path.splitext(filename)[0]}.png"
            save_full_path = os.path.join(save_dir, save_filename)
            cv2.imwrite(save_full_path, mask_expend)

    # mask = load_mask('/media/junz/4TB-1/ldh/papers/0_Infusion/test/0_get_inpaint_mask/inpaint_mask/ipt_mask_00012.png')
    # mask_expend = mask_edge_expend(mask, 20).astype(np.uint8) * 255
    #
    # cv2.imwrite('/media/junz/4TB-1/ldh/papers/0_Infusion/test/0_get_inpaint_mask/inpaint_mask/ipt_mask_00012_ed.png', mask_expend)