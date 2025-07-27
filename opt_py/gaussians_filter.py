import cv2
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
import torch
import os
from os import makedirs, path
from errno import EEXIST

import numpy as np
import argparse

max_sh_degree = 0


def load_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # xyz features_dc features_extra opacities scales rots
    return xyz, features_dc, features_extra, opacities, scales, rots


def pcd_in_mask(point_cloud, camera_matrix, extrinsic_matrix, mask, kernel_size=30):
    """返回投影在mask内的point_cloud索引."""

    point_cloud = np.asarray(point_cloud.points)
    # 获取图像的高度和宽度
    height, width = mask.shape

    mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)

    # 将点云从世界坐标系转换回相机坐标系
    extrinsic_matrix_inv = np.linalg.inv(extrinsic_matrix)
    homogeneous_world_coords = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    homogeneous_camera_coords = homogeneous_world_coords @ extrinsic_matrix_inv.T
    camera_coords = homogeneous_camera_coords[:, :3] / homogeneous_camera_coords[:, 3][:, np.newaxis]

    # 投影到图像平面
    pixels = camera_coords @ camera_matrix.T
    pixels /= pixels[:, 2][:, np.newaxis]

    point_indices = []
    for i in range(pixels.shape[0]):
        x, y, z = int(pixels[i, 0]), int(pixels[i, 1]), camera_coords[i, 2]
        if 0 <= x < width and 0 <= y < height and mask[y, x] == 1:
            point_indices.append(i)

    return np.array(point_indices)


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def convert_np_tensor(xyz, features_dc, features_rest, opacity, scaling, rotation):
    xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
    features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
    features_rest = torch.tensor(features_rest, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
    opacity = torch.tensor(opacity, dtype=torch.float, device="cuda")
    scaling = torch.tensor(scaling, dtype=torch.float, device="cuda")
    rotation = torch.tensor(rotation, dtype=torch.float, device="cuda")
    return xyz, features_dc, features_rest, opacity, scaling, rotation


def construct_list_of_attributes(features_dc, features_rest, scaling, rotation):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(features_dc.shape[1] * features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(features_rest.shape[1] * features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l


def save_ply(xyz, features_dc, features_rest, opacity, scaling, rotation, path_save):
    mkdir_p(os.path.dirname(path_save))

    xyz = xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scale = scaling.detach().cpu().numpy()
    rotation = rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in
                  construct_list_of_attributes(features_dc, features_rest, scaling, rotation)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path_save)


def process_gaussians(xyz, scales, opacities, model_path, source_path, last_iter, size):
    # 1. 根据xyz创建o3d点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # 2. 获取各视角的信息，投影计算点云投影到mask内的高斯核索引1
    file_path = f"{model_path}/apart_45_views.txt"
    all_indices = []

    with open(file_path, 'r') as file:
        for line in file.readlines():
            line = line.strip()  # 去除每行的空白字符（如换行符等）
            image_name = os.path.splitext(os.path.basename(line))[0]
            # 加载目标视角参数
            mask_path = f'{source_path}/seg/{image_name}.png'
            c2w_path = f'{model_path}/train/ours_{last_iter}/c2w/{image_name}.npy'
            intri_path = f'{model_path}/train/ours_{last_iter}/intri/{image_name}.npy'

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask[mask != 0] = 1  # 将掩码所有非0处转换为1
            c2w = np.load(c2w_path)
            intri = np.load(intri_path)
            h, w = mask.shape

            ind1 = pcd_in_mask(pcd, intri, c2w, mask, kernel_size=size)
            all_indices.append(ind1)

    # 取所有视角索引的交集
    if all_indices:
        combined_indices = all_indices[0]
        for ind in all_indices[1:]:
            combined_indices = np.intersect1d(combined_indices, ind)
            # combined_indices = np.union1d(combined_indices, ind)
    else:
        combined_indices = np.array([])

    return np.array(combined_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Gaussians.')
    parser.add_argument('--input_ply', type=str, required=True, help='The path to the Input Gaussians')
    parser.add_argument('--save_ply', type=str, required=True, help='The path to save the Gaussians.')
    parser.add_argument('--model_path', required=True, help='Path to the model.')
    parser.add_argument('--source_path', required=True, help='Path to the source scene.')
    parser.add_argument('--last_iter', required=True, type=int, help='last iteration number.')
    parser.add_argument('--size', required=True, type=int, default=40, help='size of removing gaussians.')

    args = parser.parse_args()

    input_ply = args.input_ply
    path_save = args.save_ply
    # 读取和处理点云
    # pcd, filtered_attributes = load_point_cloud(input_ply_path)
    xyz2, features_dc2, features_extra2, opacities2, scales2, rots2 = load_ply(input_ply)
    # 去除高斯核
    ind = process_gaussians(xyz2, scales2, opacities2, args.model_path, args.source_path, args.last_iter,args.size)
    indices = np.array([True] * len(xyz2))
    indices[ind] = False
    xyz_rgb = xyz2[indices]
    features_dc = features_dc2[indices]
    features_rest = features_extra2[indices]
    opacity = opacities2[indices]
    scaling = scales2[indices]
    rotation = rots2[indices]
    print("{} floaters were removed.".format(len(xyz2) - len(xyz_rgb)))

    xyz, features_dc, features_rest, opacity, scaling, rotation = convert_np_tensor(xyz_rgb, features_dc, features_rest,
                                                                                    opacity, scaling, rotation)
    save_ply(xyz, features_dc, features_rest, opacity, scaling, rotation, path_save)
