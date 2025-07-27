import argparse
from random import random

from scipy.ndimage import label
import numpy as np
import os
from glob import glob
import cv2
from plyfile import PlyData
import open3d as o3d
from scipy.ndimage import convolve
import sys



sys.path.append("../")
from compose import load_ply, convert_np_tensor
from depth_inpainting.utils.image_util import create_point_cloud


def load_mask(mask_path):
    """读取mask，并转换为0，1mask"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask[mask != 0] = 1  # 将掩码所有非0处转换为1
    return mask


def mask_edge_expend(mask, size):
    """扩大mask的边界，size为扩大的尺寸,即size个像素"""
    mask[mask>0] = 1
    mask = mask.astype(np.uint8)
    kernel = np.ones((size, size), np.uint8)
    mask_expend = cv2.dilate(mask, kernel, iterations=1)
    return mask_expend


def point_cloud_mask(points, mask):
    """只保留mask内生成的点云"""
    mask = mask.astype(bool)
    # 使用布尔索引直接筛选符合条件的点云
    return points[mask.reshape(-1)]


def point_cloud_to_2d_mask(point_cloud, camera_matrix, extrinsic_matrix, image_shape):
    """得到投影的mask."""

    # 获取图像的高度和宽度
    height, width = image_shape

    # 将点云从世界坐标系转换回相机坐标系
    extrinsic_matrix_inv = np.linalg.inv(extrinsic_matrix)
    homogeneous_world_coords = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    homogeneous_camera_coords = homogeneous_world_coords @ extrinsic_matrix_inv.T
    camera_coords = homogeneous_camera_coords[:, :3] / homogeneous_camera_coords[:, 3][:, np.newaxis]

    # 投影到图像平面
    pixels = camera_coords @ camera_matrix.T
    pixels /= pixels[:, 2][:, np.newaxis]

    count_map = np.zeros((height, width), dtype=np.float32)
    for i in range(pixels.shape[0]):
        x, y, z = int(pixels[i, 0]), int(pixels[i, 1]), camera_coords[i, 2]
        if 0 <= x < width and 0 <= y < height:
            count_map[y, x] += 1

    # 只保留点数足够多的像素
    density_threshold = 1  # 可以调整这个阈值
    proj_mask = (count_map >= density_threshold).astype(np.float32)
    proj_mask[proj_mask != 0] = 1
    return proj_mask


def load_point_cloud(ply_url):
    """读取ply点云文件，返回点云对象."""
    plydata = PlyData.read(ply_url)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    r = plydata['vertex']['red']
    g = plydata['vertex']['green']
    b = plydata['vertex']['blue']
    points = np.column_stack([x, y, z])
    colors = np.column_stack([r, g, b])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    normalized_colors = colors / 255.0
    pcd.colors = o3d.utility.Vector3dVector(normalized_colors)
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=4.0)
    # pcd = pcd.select_by_index(ind)

    return pcd


def mask_filter(mask, filter_size, threshold):
    """输入一个mask和滤波器的大小filter_size * filter_size，据阈值threshold输出一个经过滤波增强的mask"""
    # 创建一个滤波器, 内容全1, 大小为 filter_size * filter_size 的矩阵
    filter_kernel = np.ones((filter_size, filter_size))
    # 将 mask 中的非零值全部置1
    binary_mask = (mask != 0).astype(int)
    # 计算填充数量
    pad_size = filter_size // 2  # '//' 整除
    # 对掩码进行填充
    padded_mask = np.pad(binary_mask, pad_size, mode='constant', constant_values=0)
    # 使用滤波器对填充后的 mask 进行卷积
    filtered_mask = convolve(padded_mask, filter_kernel, mode='constant', cval=0.0)
    # 裁剪结果，使其与输入掩码大小相同
    filtered_mask = filtered_mask[pad_size:-pad_size, pad_size:-pad_size]
    # 归一化卷积结果，使其在0到1之间
    max_value = filter_size * filter_size
    normalized_mask = filtered_mask / max_value
    # 根据设置的阈值，低于阈值的置零，高于阈值的置一
    enhanced_mask = (normalized_mask > threshold).astype(int)

    return enhanced_mask


def process_2d_density_mask(mask, n, threshold):
    """根据网格内密度使mask不再是稀疏的点"""
    # 获取图像的高和宽
    height, width = mask.shape
    # 创建一个新的图像用于存储结果
    result_mask = mask.copy()
    # 遍历每个nxn像素的网格
    for i in range(0, height, 4):
        for j in range(0, width, 4):
            # 获取当前nxn网格
            grid = mask[i:i + n, j:j + n]
            # 计算非0像素的数量
            non_zero_count = np.count_nonzero(grid)
            # 如果非0像素的数量超过阈值，则将网格内的所有像素变为全白
            if non_zero_count > threshold:
                result_mask[i:i + n, j:j + n] = 1
            else:
                result_mask[i:i + n, j:j + n] = 0

    return result_mask


# def get_neighbor_image_names(image_name, offset, total):
#     """得到两侧视角的名字"""
#     non_number = ''.join([c for c in image_name if not c.isdigit()])
#     number = ''.join([c for c in image_name if c.isdigit()])
#     number_digits = len(number)
#
#     int_number = int(number)
#     left_number = max(1, int_number - offset)
#     right_number = min(total - 1, int_number + offset)
#
#     left_number_str = str(left_number).zfill(number_digits)
#     right_number_str = str(right_number).zfill(number_digits)
#
#     return non_number + left_number_str, non_number + right_number_str


def read_ply_xyz(path):
    plydata = PlyData.read(path)
    vertex_data = plydata['vertex']

    xyz = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T

    point_cloud = np.array(xyz)

    return point_cloud


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to the model.')
    parser.add_argument('--source_path', required=True, help='Path to the source.')
    # parser.add_argument('--target_image_name', required=True, help='Name of the target image.')
    parser.add_argument('--last_iter', required=True, type=int, help='last iteration number.')
    parser.add_argument('--expand_size', type=int, default=0, help='expand size')
    args = parser.parse_args()
    model_path = args.model_path
    source_path = args.source_path
    # image_name = args.target_image_name
    last_iter = args.last_iter
    expand_size = args.expand_size

    path_to_ply_file = f"{model_path}/point_cloud/iteration_{last_iter}/point_cloud.ply"
    # merged_point_cloud_side = read_ply_xyz(path_to_ply_file)
    xyz2, features_dc2, features_extra2, opacities2, scales2, rots2 = load_ply(path_to_ply_file)
    merged_point_cloud_side = xyz2

    # 获取目录下文件个数，作为循环次数
    render_dir = f'{model_path}/train/ours_{last_iter}/renders/'
    file_count = len([name for name in os.listdir(render_dir) if os.path.isfile(os.path.join(render_dir, name))])
    if not os.path.exists(f"{model_path}/mask/sub_mask"):
        os.makedirs(f"{model_path}/mask/sub_mask")

    for i in range(file_count):
        image_name = os.path.splitext(os.path.basename(os.listdir(render_dir)[i]))[0]

        # 加载目标视角参数
        mask_path = f'{model_path}/mask/seg_expand/{image_name}.png'
        depth_path = f'{model_path}/train/ours_{last_iter}/depth/{image_name}.npy'
        c2w_path = f'{model_path}/train/ours_{last_iter}/c2w/{image_name}.npy'
        intri_path = f'{model_path}/train/ours_{last_iter}/intri/{image_name}.npy'

        mask = load_mask(mask_path)
        depth = np.load(depth_path)
        c2w = np.load(c2w_path)
        intri = np.load(intri_path)
        h, w = mask.shape

        # 将点云投影到目标视角，得到二维 mask
        proj_mask_side_all = point_cloud_to_2d_mask(merged_point_cloud_side, intri, c2w, (h, w))
        # cv2.imwrite(f'{model_path}/mask/sub_mask/{image_name}_point.png', proj_mask_side_all*255)

        # 对proj_mask_side_all 做过滤处理，之后取反？
        filter_size = 3
        filter_thr = 0.2  # 阈值越大，mask区域越小，最后修复mask越大
        enhanced_mask1 = mask_filter(proj_mask_side_all, filter_size, filter_thr)
        # cv2.imwrite(f'{model_path}/mask/sub_mask/{image_name}_enhanced_mask1.png', enhanced_mask1 * 255)
        enhanced_mask2 = mask_filter(enhanced_mask1, 9, 0.2)
        # cv2.imwrite(f'{model_path}/mask/sub_mask/{image_name}_enhanced_mask2.png', enhanced_mask2 * 255)

        # block_size = 16
        # density_thr = block_size * block_size * 3 / 4  # 阈值越大，mask区域越小， 最后修复mask越大
        # density_mask_side = process_2d_density_mask(enhanced_mask_side, block_size, density_thr)

        combined_mask_side_inv = cv2.bitwise_not(enhanced_mask2.astype(np.uint8) * 255) * mask
        # cv2.imwrite(f'{model_path}/mask/sub_mask/{image_name}_inv.png',
        #             combined_mask_side_inv)

        # 进行连通组件分析, 去除离群的区域
        combined_mask_side_inv[combined_mask_side_inv != 0] = 1
        labeled_mask, num_features = label(combined_mask_side_inv)
        # 计算每个连通区域的面积
        sizes = np.bincount(labeled_mask.flatten())[1:]
        # 找到最大面积的索引
        if len(sizes) > 0:
            max_area_index = np.argmax(sizes) + 1
        else:
            # 没有找到可用于计算的区域面积，使用原始mask
            max_area_index = 0  # 默认值，根据实际需求调整
            src_mask_path=f'{source_path}/seg/{image_name}.png'
            src_mask=load_mask(src_mask_path)
            kernel_size = 10
            src_mask = cv2.dilate(src_mask, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)
            cv2.imwrite(f'{model_path}/mask/sub_mask/{image_name}.png', src_mask * 255)
            continue

        # 创建只保留最大面积区域的新 mask
        mask_cleaned = (labeled_mask == max_area_index).astype(np.uint8)
        mask_cleaned[mask_cleaned == 1] = 255

        # cv2.imwrite(f'{model_path}/mask/sub_mask/{image_name}_cleaned.png', mask_cleaned)

        # 再过滤一次，让边缘变得光滑
        mask_cleaned = mask_filter(mask_cleaned, 15, 0.2)
        mask_cleaned[mask_cleaned == 1] = 255
        # cv2.imwrite(f'{model_path}/mask/sub_mask/{image_name}_cleaned_enhance.png', mask_cleaned * 255)

        # 扩大
        expand_mask = mask_edge_expend(mask_cleaned, expand_size)
        cv2.imwrite(f'{model_path}/mask/sub_mask/{image_name}.png', expand_mask * 255)

