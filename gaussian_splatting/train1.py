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
from datetime import datetime
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
import torchvision
import os
from PIL import Image
import torch
from random import randint

from scene.uncertainty_model import UncertaintyModel
from utils.weight_utils import get_dynamic_weight
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import lpips
from utils.general_utils import PILtoTorch

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             load_iteration, mask_training, color_aug, main_image):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    if load_iteration != 0:
        scene = Scene(dataset, gaussians, load_iteration)
    else:
        scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    LPIPS = lpips.LPIPS(net='vgg')
    # LPIPS.eval()
    for param in LPIPS.parameters():
        param.requires_grad = False

    LPIPS = LPIPS.to(torch.device("cuda:0"))

    # 修复视角的图片名称
    ipt_images = []
    for filename in os.listdir(args.inpaint_dir):
        # for filename in os.listdir(f"{model_path}/inpainted_images_dir"):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            # 得到每一个修复图像的路径
            file_name = os.path.splitext(filename)[0]
            ipt_images.append(file_name)

    # 创建 UncertaintyModel 对象
    if not os.path.exists(f'{args.model_path}/uncertainty_params/'):
        os.makedirs(f'{args.model_path}/uncertainty_params/')
    uncertainty_models = []
    for cam in scene.getTrainCameras():
        if cam.image_name in ipt_images:
            # if cam.image_name in ipt_images and cam.image_name != "frame_00033":
            uncertainty_model = UncertaintyModel(cam, args.model_path)
            # 初始化uncertainty
            uncertainty_model.uncertainty_init(model_path=args.model_path, depth_dir=args.depth_dir)
            torch.save(uncertainty_model.get_uncertainty().data,
                       f'{args.model_path}/uncertainty_params/{uncertainty_model.camera.image_name}_uncertainty_init.pth')
            uncertainty_models.append(uncertainty_model)

    # 创建优化器
    all_uncertainty_params = []
    for model in uncertainty_models:
        if hasattr(model, 'uncertainty'):
            all_uncertainty_params.append(model.uncertainty)
    uncertainty_optimizer = torch.optim.Adam(all_uncertainty_params, lr=0.02)

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        #     if inpainting_dir and os.listdir(inpainting_dir):
        #         viewpoint_stack_ipt = viewpoint_stack    #用viewpoint_stack_ipt来存放修复图像的视角，在viewpoint_stack进行选择
        # viewpoint_cam = viewpoint_stack_ipt.pop(randint(0, len(viewpoint_stack) - 1))
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, random_color=color_aug)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        depth = render_pkg["depth_3dgs"]  # 当前视角渲染的深度

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        model_path = dataset.model_path  # 数据集的输出路径
        image_name = viewpoint_cam.image_name  # 取出或名字

        if mask_training:
            kernel_size = 10
            image_mask = cv2.dilate(viewpoint_cam.original_image_mask,
                                    np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)

            image_m = image * torch.tensor(1 - image_mask).cuda().repeat(3, 1, 1)
            gt_image_m = gt_image * torch.tensor(1 - image_mask).cuda().repeat(3, 1, 1)

            Ll1 = l1_loss(image_m, gt_image_m, mask=viewpoint_cam.original_image_mask)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_m, gt_image_m))

            loss.backward()

        else:
            kernel_size = 10
            image_mask = cv2.dilate(viewpoint_cam.original_image_mask,
                                    np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)

            # # mask外的图像
            # image_m = image * torch.tensor(1 - image_mask).cuda().repeat(3, 1, 1)
            # gt_image_m = gt_image * torch.tensor(1 - image_mask).cuda().repeat(3, 1, 1)

            sub_mask_path = f"{model_path}/mask/sub_mask/{image_name}.png"
            # sub_mask_path = f"{model_path}/mask/seg_expand_ablation/{image_name}.png"
            if os.path.exists(sub_mask_path):
                sub_mask = cv2.imread(sub_mask_path, 0)
            else:
                sub_mask = image_mask

            sub_mask[sub_mask != 0] = 1
            # mask外的图像
            image_m = image * torch.tensor(1 - sub_mask).cuda().repeat(3, 1, 1)
            gt_image_m = gt_image * torch.tensor(1 - sub_mask).cuda().repeat(3, 1, 1)

            if image_name in ipt_images:  # 修复视角参考整张图像作loss, 加入权重图

                gt_depth = np.load(f"{model_path}/depth_completed/depth_completed_{image_name}/{image_name}_depth.npy")
                gt_depth = torch.from_numpy(gt_depth).cuda()

                # 动态权重，开始权重大，后续权重逐渐减少
                w = get_dynamic_weight(iteration, args.iterations)

                # Ll1 = l1_loss(image, gt_image, mask=viewpoint_cam.original_image_mask)
                # loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))) * w

                if image_name in main_image:
                    Ll1 = l1_loss(image, gt_image, mask=viewpoint_cam.original_image_mask)
                    loss_render = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                    loss_lpips = LPIPS(image, gt_image)
                    # loss_render = loss_render * w + 0.5 * loss_lpips
                    loss_render = loss_render * w + 0.5 * loss_lpips + 0.5 * l1_loss(depth, gt_depth)
                else:
                    Ll1 = l1_loss(image_m, gt_image_m, mask=viewpoint_cam.original_image_mask)
                    loss_render = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_m, gt_image_m))
                                   + 0.5 * l1_loss(depth, gt_depth))
                    # loss_render = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_m, gt_image_m))


                Ll1_uncertainty = 0
                if iteration > opt.iterations * 0.25:
                    for uncertainty_model in uncertainty_models:
                        if uncertainty_model.camera.image_name == image_name:
                            Ll1_uncertainty = uncertainty_model.calculate_uncertainty_loss(image, gt_image)
                            break
                # Ll1 = l1_loss(image_c, gt_image_c, mask=viewpoint_cam.original_image_mask)
                # loss_render = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_c, gt_image_c))

                loss = Ll1_uncertainty + loss_render
                # loss = loss_render + (uncertainty * uncertainty_mask).mean() * 0.01

                # loss_lpips = LPIPS(image, gt_image)
                # loss = loss + loss_lpips * 0.1

            # 如果是其他视角，则只对mask外部分作loss
            else:
                Ll1 = l1_loss(image_m, gt_image_m, mask=viewpoint_cam.original_image_mask)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_m, gt_image_m))

            loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Densification

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])

                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # if iteration % 100 == 0:
            #     gaussians.remove_large_points(scene.cameras_extent)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                uncertainty_optimizer.step()
                uncertainty_optimizer.zero_grad()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


    for model in uncertainty_models:
        uncertainty = model.uncertainty_sample(model.get_uncertainty(), is_up=True) * model.gradient_mask
        uncertainty = uncertainty.data.cpu().numpy()
        # 提取非零值
        non_zero_mask = uncertainty != 0
        non_zero_values = uncertainty[non_zero_mask]
        if non_zero_values.size > 0:
            # 对非零值进行归一化
            min_val = non_zero_values.min()
            max_val = non_zero_values.max()
            normalized_non_zero = (non_zero_values - min_val) / (max_val - min_val)
            uncertainty[non_zero_mask] = normalized_non_zero
        # 将归一化后的数据转换为0 - 255的8位无符号整数
        uncertainty = (uncertainty * 255.0).astype(np.uint8)
        # 应用颜色映射
        uncertainty = cv2.applyColorMap(uncertainty, cv2.COLORMAP_JET)

        output_path = f'{args.model_path}/uncertainty_params/{model.camera.image_name}_uncertainty.png'
        cv2.imwrite(output_path, uncertainty)
        # # 保存不确定性参数
        # torch.save(model.get_uncertainty().data,
        #            f'{args.model_path}/uncertainty_params/{model.camera.image_name}_uncertainty.pth')


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./result/bear", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # update parameters for these three
    parser.add_argument('--ip', type=str, default="127.0.0.212")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--load_iteration', type=int, default=0)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--mask_training', action='store_true', default=False)
    parser.add_argument('--color_aug', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # 添加新的参数
    parser.add_argument('--depth_dir', type=str, default="depth_completed",
                        help="Directory containing depth images")
    parser.add_argument("--inpaint_dir", type=str, default=None,
                        help="Directory containing inpainting images for loss calculation")
    parser.add_argument("--main_image", type=str, nargs='+', default=None,
                        help="训练时最先参与训练的主视角（可传入多个值，组成列表）")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.load_iteration, args.mask_training, args.color_aug)
    # 传入inpainting_dir参数
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.load_iteration,
             args.mask_training, args.color_aug, args.main_image)
    # All done
    print("\nTraining complete.")
