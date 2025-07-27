import os
import argparse

import cv2
import numpy as np
from skimage.io import imread
from open3d import PointCloud, registration_icp, TransformationEstimationPointToPoint


def get_image_names(input_rgb_dir):
    image_names = []
    for root, dirs, files in os.walk(input_rgb_dir):
        for file in files:
            if file.endswith('.npy'):
                image_name = os.path.splitext(file)[0]
                image_names.append(image_name)
    return image_names


def read_data(image_name, input_mask_dir, input_depth_dir, c2w_dir, intri_dir):
    mask_path = os.path.join(input_mask_dir, f"{image_name}.png")
    depth_path = os.path.join(input_depth_dir.format(image_name=image_name), f"{image_name}_depth.npy")
    c2w_path = os.path.join(c2w_dir, f"{image_name}.npy")
    intri_path = os.path.join(intri_dir, f"{image_name}.npy")

    mask_data = imread(mask_path)
    depth_data = np.load(depth_path)
    c2w_data = np.load(c2w_path)
    intri_data = np.load(intri_path)

    return mask_data, depth_data, c2w_data, intri_data


def depth_to_point_cloud(depth, mask, c2w, intri):
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()
    valid_mask = mask.flatten() == 255
    u = u[valid_mask]
    v = v[valid_mask]
    depth_valid = depth.flatten()[valid_mask]

    # 从深度图和相机内参计算点云在相机坐标系下的坐标
    fx, fy = intri[0, 0], intri[1, 1]
    cx, cy = intri[0, 2], intri[1, 2]
    x = (u - cx) * depth_valid / fx
    y = (v - cy) * depth_valid / fy
    z = depth_valid
    points_camera = np.vstack((x, y, z)).T

    # 转换到世界坐标系
    points_world = np.dot(c2w[:3, :3], points_camera.T).T + c2w[:3, 3]

    pcd = PointCloud()
    pcd.points = points_world
    return pcd


def align_depths_with_icp(depth_datas, mask_datas, c2w_datas, intri_datas):
    # 选择第一个视角作为主视角
    main_pcd = depth_to_point_cloud(depth_datas[0], mask_datas[0], c2w_datas[0], intri_datas[0])
    aligned_depths = [depth_datas[0]]

    for i in range(1, len(depth_datas)):
        current_pcd = depth_to_point_cloud(depth_datas[i], mask_datas[i], c2w_datas[i], intri_datas[i])
        result_icp = registration_icp(current_pcd, main_pcd, 0.1, np.eye(4),
                                      TransformationEstimationPointToPoint())
        transformation = result_icp.transformation

        # 将变换应用到当前深度数据对应的点云，这里简单将点云转换回深度数据
        # 实际中可能需要更复杂的处理
        points_camera_current = np.array(current_pcd.points)
        points_world_current = np.dot(transformation[:3, :3], points_camera_current.T).T + transformation[:3, 3]
        # 这里假设能通过逆投影等操作将点云转换回深度数据，实际可能需要更复杂的处理
        # 简单示例忽略
        aligned_depth = depth_datas[i]
        aligned_depths.append(aligned_depth)

    return aligned_depths


def main():
    parser = argparse.ArgumentParser(description='Align multi-view depths using ICP')
    parser.add_argument('--input_rgb_dir', type=str, required=True, help='Input RGB images directory')
    parser.add_argument('--input_mask_dir', type=str, required=True, help='Input mask images directory')
    parser.add_argument('--input_depth_dir', type=str, required=True, help='Input depth images directory')
    parser.add_argument('--c2w_dir', type=str, required=True, help='c2w data directory')
    parser.add_argument('--intri_dir', type=str, required=True, help='intri data directory')

    args = parser.parse_args()

    image_names = get_image_names(args.input_rgb_dir)

    all_rgb_datas = []
    all_mask_datas = []
    all_depth_datas = []
    all_c2w_datas = []
    all_intri_datas = []

    for image_name in image_names:
        mask_data, depth_data, c2w_data, intri_data = read_data(image_name,args.input_mask_dir,
                                                                args.input_depth_dir,
                                                                args.c2w_dir, args.intri_dir)
        all_mask_datas.append(mask_data)
        all_depth_datas.append(depth_data)
        all_c2w_datas.append(c2w_data)
        all_intri_datas.append(intri_data)

    aligned_depths = align_depths_with_icp(all_depth_datas, all_mask_datas, all_c2w_datas, all_intri_datas)

    # 这里可以添加保存对齐后深度数据的代码
    for i, aligned_depth in enumerate(aligned_depths):
        save_path = os.path.join(args.input_depth_dir.format(image_name=image_names[i]),
                                 f"{image_names[i]}_aligned_depth.npy")
        save_colored_path = os.path.join(args.input_depth_dir.format(image_name=image_names[i]),
                                 f"{image_names[i]}_aligned_depth.png")
        np.save(save_path, aligned_depth)
        save_depth_as_colorful_image(aligned_depth, save_colored_path)


def save_depth_as_colorful_image(depth, output_path):
    img = depth
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255.0).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, img)

if __name__ == "__main__":
    main()
