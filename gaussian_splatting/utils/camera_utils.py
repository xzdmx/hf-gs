#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import re

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from PIL import Image

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    gt_image = resized_image_rgb[:3, ...]

    resized_image_mask_rgb = cam_info.image_mask.resize(resolution)
    gt_image_mask = np.array(resized_image_mask_rgb)
    gt_image_mask = np.where(gt_image_mask > 127, 1, 0).astype(np.uint8)

    loaded_mask = None
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, image_mask=gt_image_mask, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []
    id = 0
    if args.unique_image !='nothing':
        for id, c in enumerate(cam_infos):
            if c.image_path.split('/')[-1] == args.unique_image:
                if args.new_image!="image":
                    c=c._replace(image_path=args.new_image)
                    c=c._replace(image=Image.open(args.new_image).convert("RGB"))
                camera_list.append(loadCam(args, id, c, resolution_scale))
                id+=1

    # args.inpaint_dir若不为空，则取这个目录下的图片的视角
    elif args.inpaint_dir != None:
        # 用于存储找到的图片路径
        ipt_images = []
        ipt_image_names = []
        for filename in os.listdir(args.inpaint_dir):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                # 得到每一个修复图像的路径
                file_path = os.path.join(args.inpaint_dir, filename)
                ipt_image_name = os.path.splitext(file_path.split('/')[-1])[0]
                ipt_images.append(file_path)
                ipt_image_names.append(ipt_image_name)

        file_path = f"{args.model_path}/apart_45_views.txt"
        frames=[]
        with open(file_path, 'r') as file:
            for line in file.readlines():
                line = line.strip()  # 去除每行的空白字符（如换行符等）
                image_name = os.path.splitext(os.path.basename(line))[0]
                frames.append(image_name)

        # for id, c in enumerate(cam_infos):
        #     img_name = os.path.splitext(c.image_path.split('/')[-1])[0]
        #     for frame in frames:
        #         frame_name = os.path.splitext(frame.split('/')[-1])[0]
        #         if img_name == frame_name:
        #             for ipt_image in ipt_images:
        #                 ipt_image_name = os.path.splitext(ipt_image.split('/')[-1])[0]
        #                 if frame_name == ipt_image_name:
        #                     if args.whole_training: # 最后一起训练时使用不确定性训练后渲染的结果
        #                         c = c._replace(image_path=ipt_image)
        #                         c = c._replace(image=Image.open(ipt_image).convert("RGB"))
        #                     else:
        #                         c = c._replace(image_path=ipt_image)
        #                         c = c._replace(image=Image.open(ipt_image).convert("RGB"))
        #                     break
        #             camera_list.append(loadCam(args, id, c, resolution_scale))
        #         else:
        #             continue

        for id, c in enumerate(cam_infos):
            img_name = os.path.splitext(c.image_path.split('/')[-1])[0]
            for frame in frames:
                frame_name = os.path.splitext(frame.split('/')[-1])[0]
                if img_name == frame_name:
                    # 换成ours_1中render的图像路径（不是修复图像，但是关键帧）
                    if frame_name not in ipt_image_names:
                        path = f'{args.model_path}/train/ours_1/renders/{frame_name}.png'
                        # path = f'{args.model_path}/train/ours_30000/renders/{frame_name}.png' # spin-nerf dataset
                        c = c._replace(image_path=path)
                        c = c._replace(image=Image.open(path).convert("RGB"))
                        camera_list.append(loadCam(args, id, c, resolution_scale))
                        break

            for ipt_image in ipt_images:
                ipt_image_name = os.path.splitext(ipt_image.split('/')[-1])[0]
                if img_name == ipt_image_name:
                    c = c._replace(image_path=ipt_image)
                    c = c._replace(image=Image.open(ipt_image).convert("RGB"))
                    camera_list.append(loadCam(args, id, c, resolution_scale))
                    break

    else:
        for id, c in enumerate(cam_infos):
            camera_list.append(loadCam(args, id, c, resolution_scale))
    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
