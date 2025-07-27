import argparse
import glob
import os
import numpy as np


def load_camera_params(filename):
    return np.load(filename)


def get_camera_direction(c2w):
    # 假设相机在原点看向 -z 方向，相机的方向向量是从世界坐标系原点指向相机位置
    return c2w[:3, 3]


def calculate_angle_difference(c2w1, c2w2):
    v1 = get_camera_direction(c2w1)
    v2 = get_camera_direction(c2w2)
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    if cos_theta < -1:
        cos_theta = -1
    elif cos_theta > 1:
        cos_theta = 1
    return np.arccos(cos_theta) * 180 / np.pi


def find_next_camera_with_approx_angle(start_camera, all_cameras, target_angle=45, tolerance=5):
    start_c2w = load_camera_params(start_camera)
    for camera in all_cameras:
        if camera == start_camera:
            continue
        c2w = load_camera_params(camera)
        angle_diff = calculate_angle_difference(start_c2w, c2w)
        if target_angle - tolerance <= angle_diff <= target_angle + tolerance:
            return camera
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--c2w_path', required=True, help='Path to the c2w files')
    parser.add_argument('--save_path', required=True, help='Path to save the selected views file')
    args = parser.parse_args()

    all_c2w_files = sorted(glob.glob(os.path.join(args.c2w_path, '*.npy')))

    selected_files = []
    start_camera = all_c2w_files[0]
    selected_files.append(os.path.splitext(os.path.basename(start_camera))[0])
    remaining_cameras = all_c2w_files[1:]

    while len(selected_files) < 8:  # 因为 360°/45° = 8
        next_camera = find_next_camera_with_approx_angle(start_camera, remaining_cameras)
        if next_camera:
            next_filename = os.path.splitext(os.path.basename(next_camera))[0]
            if next_filename not in selected_files:
                selected_files.append(next_filename)
                start_camera = next_camera
                remaining_cameras = [c for c in remaining_cameras if os.path.splitext(os.path.basename(c))[0] > next_filename]
        else:
            # 如果找不到满足条件的，就从剩下的文件中随机选择一个作为下一个起始点
            if remaining_cameras:
                start_camera = remaining_cameras[0]
                remaining_cameras = remaining_cameras[1:]
            else:
                break

    with open(args.save_path, 'w') as f:
        for filename in selected_files:
            f.write(f'{filename}\n')

    print(f"The selected camera files have been saved to {args.save_path}.")