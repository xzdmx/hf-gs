import os

import cv2
import math

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from scene.cameras import Camera


class UncertaintyModel(nn.Module):

    def __init__(self, camera: Camera, model_path):
        super().__init__()

        self.model_path=model_path
        self.camera = camera
        self.down_sample_scale = 8

        # 初始化不确定性参数，形状与相机图片相同
        self.uncertainty = nn.Parameter(
            torch.zeros(
                (self.camera.image_height // self.down_sample_scale, self.camera.image_width // self.down_sample_scale),
                device=self.camera.data_device,
                requires_grad=True)
        )  # 下采样防止内存溢出

        # 将 numpy 数组转换为 torch.Tensor 并设置为参数
        # 类初始化时添加model_path
        sub_mask_path = f"{self.model_path}/mask/sub_mask/{self.camera.image_name}.png"
        if os.path.exists(sub_mask_path):
            sub_mask = cv2.imread(sub_mask_path, 0)
        else:
            sub_mask = self.camera.original_image_mask
        sub_mask[sub_mask != 0] = 1
        self.gradient_mask = torch.from_numpy(sub_mask).to(self.camera.data_device)

    def get_uncertainty(self):
        """
        返回经过掩码处理后的不确定性值
        """
        # 应用 ReLU 函数确保不确定性值非负，同时保留原本为 0 的值不变
        positive_uncertainty = F.relu(self.uncertainty)

        gradient_mask = self.gradient_mask.clone()
        if gradient_mask.shape != positive_uncertainty.shape:
            # 对positive_uncertainty上采样
            positive_uncertainty = self.uncertainty_sample(positive_uncertainty, is_up=True)

        # 对不确定性值应用掩码
        masked_uncertainty = positive_uncertainty * gradient_mask

        # 对masked_uncertainty下采样
        masked_uncertainty = self.uncertainty_sample(masked_uncertainty, is_down=True)

        # 对 masked_uncertainty 进行限制
        clamped_uncertainty = torch.clamp(masked_uncertainty, min=0.01)
        del gradient_mask

        return clamped_uncertainty

    def get_gradient_mask(self):
        """
        返回当前的梯度掩码
        """
        return self.gradient_mask

    def get_uncertainty_image(self, image):
        """
        返回不确定性形式的图像
        """
        # 上采样
        uncertainty = (self.uncertainty_sample(self.get_uncertainty(), is_up=True) * self.gradient_mask).repeat(3, 1, 1)

        # 确保图像和不确定性的形状匹配
        assert image.shape == uncertainty.shape

        confidence = torch.exp(-uncertainty).type(image.dtype)

        image_uncertainty = confidence * image
        return image_uncertainty

    def calculate_uncertainty_loss(self, image, gt_image):
        """
        返回不确定性形式的图像间的损失
        """
        # 上采样
        uncertainty = self.uncertainty_sample(self.get_uncertainty(), is_up=True).repeat(3, 1, 1)

        image_m = image * self.gradient_mask
        gt_image_m = gt_image * self.gradient_mask
        # 确保图像和不确定性的形状匹配
        assert image.shape == gt_image.shape == uncertainty.shape

        # # 逐像素计算 L1 损失
        # l1_loss_per_pixel = torch.abs(image_m - gt_image_m)
        # 逐像素计算L2损失
        l2_loss_per_pixel = torch.pow((image_m - gt_image_m), 2)
        # 假设式子中的其他参数为 lambda_value 和 yio_value，你需要根据实际情况设置它们的值
        lambda_value = 1.0
        # lambda_value = 0.0 # 测试去掉正则项

        # 只对 self.gradient_mask 区域中的 uncertainty 计算 log
        uncertainty_masked = torch.where(self.gradient_mask > 0, uncertainty, torch.tensor(1.0).cuda())
        log_uncertainty_masked = torch.where(self.gradient_mask > 0, torch.log(uncertainty),
                                             torch.tensor(0.0).cuda())
        # # 将 log_uncertainty_masked 中为 nan 的值（极大值）变为 1.0
        # log_uncertainty_masked = torch.nan_to_num(log_uncertainty_masked, nan=1.0)

        loss = (l2_loss_per_pixel / (2 * uncertainty_masked.pow(2)) +
                lambda_value * log_uncertainty_masked + torch.log(torch.tensor(2 * math.pi)) / 2).mean()
        # 逐像素计算损失
        return loss

    def uncertainty_init(self, model_path, depth_dir):
        """
        根据深度初始化mask内的不确定性
        :return:
        """
        image_name = self.camera.image_name
        depth = np.load(f"{model_path}/{depth_dir}/depth_completed_{image_name}/{image_name}_depth.npy")
        # depth = np.load(f"{model_path}/depth_completed_lama_2/depth_completed_{image_name}/{image_name}_depth.npy")
        # 获取梯度掩码内部区域的深度最小值和最大值
        mask = self.gradient_mask.clone().detach().cpu().numpy()
        mask_depth = depth * mask
        mask_depth_min = mask_depth[mask_depth > 0].min()
        mask_depth_max = mask_depth[mask_depth > 0].max()
        # 只对梯度掩码内部区域进行归一化
        depth_normalized = np.where(mask > 0, 100 * (depth - mask_depth_min) / (mask_depth_max - mask_depth_min), 0)
        depth_tensor = torch.tensor(depth_normalized).cuda()

        # uncertainty 初始化
        uncertainty_init_value = depth_tensor
        # # uncertainty 初始化
        # uncertainty_init_value = torch.ones_like(depth_tensor) * 10

        # 下采样恢复原来形状（如果需要）
        downsampled_value = self.uncertainty_sample(uncertainty_init_value, is_down=True)

        self.uncertainty = nn.Parameter(downsampled_value, requires_grad=True)

    def uncertainty_sample(self, uncertainty, is_up=False, is_down=False):
        if is_up:
            upsampled_uncertainty = F.interpolate(uncertainty.unsqueeze(0).unsqueeze(0),
                                                  size=(self.camera.image_height, self.camera.image_width),
                                                  mode='nearest')
            return upsampled_uncertainty.squeeze(0).squeeze(0)
        if is_down:
            downsampled_uncertainty = F.avg_pool2d(uncertainty.unsqueeze(0).unsqueeze(0),
                                                   kernel_size=self.down_sample_scale)
            return downsampled_uncertainty.squeeze(0).squeeze(0)
        return uncertainty

    def update_gradient_mask(self):
        """
            基于不确定性更新掩码及不确定性相关设置
        """
        # 上采样
        uncertainty = self.uncertainty_sample(self.get_uncertainty(), is_up=True)

        # 只考虑掩码区域内的不确定性
        mask_uncertainty = uncertainty[self.gradient_mask.bool()]

        # 获取从大到小排序后处于第40%位置的置信度的值
        uncertainty_40_percent, _ = torch.kthvalue(mask_uncertainty, int(len(mask_uncertainty) * 0.4))

        # 更新mask，保持类型为tensor
        new_mask = self.gradient_mask & (uncertainty <= uncertainty_40_percent)

        # 直接更新gradient_mask
        self.gradient_mask = new_mask

    def adjust_uncertainty(self, iteration, total_iterations, imgs_count):
        """
        根据迭代次数调整不确定性的值
        """
        uncertainty = self.uncertainty_sample(self.get_uncertainty(), is_up=True)
        mask = self.get_gradient_mask()

        if iteration <= total_iterations // 10:
            return  # 在前十分之一迭代次数内，不做调整

        elif iteration <= 6 * total_iterations // 10:
            # 在十分之一到第十分之六迭代次数内，对不确定性进行处理
            masked_uncertainty = uncertainty[mask.bool()].view(-1)

            step_size = ((6 * total_iterations // 10 - total_iterations // 10) // 8)
            begin_step = total_iterations // 10
            if (iteration - begin_step) % step_size < imgs_count:
                sorted_uncertainty, indices = torch.sort(masked_uncertainty)
                num_to_set = int(len(sorted_uncertainty) * (begin_step // step_size * 0.1))
                set_indices = indices[:num_to_set]
                masked_uncertainty[set_indices] = torch.finfo(torch.float32).max

                uncertainty[mask.bool()] = masked_uncertainty.view(uncertainty[mask.bool()].shape)

                # 对uncertainty下采样
                uncertainty = self.uncertainty_sample(uncertainty, is_down=True)

                self.uncertainty.data = uncertainty

        elif iteration > 6 * total_iterations // 10:
            self.uncertainty.requires_grad = False
            return  # 在最后的十分之四迭代次数内，不做调整

    def adjust_uncertainty_2(self, iteration, total_iterations, img_count):
        """
        根据迭代次数调整不确定性的值
        """
        uncertainty = self.uncertainty_sample(self.get_uncertainty(), is_up=True)
        mask = self.get_gradient_mask()

        if iteration <= 0.1 * total_iterations:
            return  # 在前十分之一迭代次数内，不做调整

        else:
            masked_uncertainty = uncertainty[mask.bool()].view(-1)

            percentage_to_set_each_time = 0.05  # 每次设置前百分之5的不确定性值为较大值
            total_percentage_to_set = 1 - (100 / img_count)  # 每个视角总共要设置为极大值的百分比
            num_operations = total_percentage_to_set / percentage_to_set_each_time
            step_size = (total_iterations - 0.1 * total_iterations) / num_operations

            begin_step = 0.1 * total_iterations
            if (iteration - begin_step) % step_size < img_count and (iteration - begin_step) > step_size:
                sorted_uncertainty, indices = torch.sort(masked_uncertainty)
                num_to_set = int(len(sorted_uncertainty) * percentage_to_set_each_time)
                set_indices = indices[:num_to_set]
                masked_uncertainty[set_indices] = torch.finfo(torch.float32).max

                # # 获取设置为极大值的区域在原始uncertainty中的索引
                # set_indices_in_uncertainty = torch.nonzero(uncertainty[mask.bool()][set_indices]).squeeze()

                uncertainty[mask.bool()] = masked_uncertainty.view(uncertainty[mask.bool()].shape)

                # 对uncertainty下采样
                uncertainty = self.uncertainty_sample(uncertainty, is_down=True)

                self.uncertainty.data = uncertainty
                # # 将设置为极大值的区域对应的参数requires_grad设置为false
                # self.uncertainty.data[set_indices_in_uncertainty] = self.uncertainty.data[set_indices_in_uncertainty].detach()


    def final_uncertainty_np(self):
        """
        返回numpy形式的最终不确定性
        """
        # 上采样
        uncertainty = self.uncertainty_sample(self.get_uncertainty(), is_up=True)
        return uncertainty.cpu().numpy()


    # def get_confidence(self):
    #     """
    #     Returns the confidence values (inverse of uncertainty)
    #     """
    #     return torch.exp(-self.get_uncertainty())
    #
    # def update_mask_by_confidence(self):
    #     """
    #     基于置信度更新掩码及不确定性相关设置
    #     """
    #     # 获取当前置信度
    #     confidence = self.get_confidence()
    #
    #     # 只考虑掩码区域内的置信度
    #     mask_confidence = confidence[self.gradient_mask.bool()]
    #
    #     # 获取从大到小排序后处于第40%位置的置信度的值
    #     confidence_40_percent, _ = torch.kthvalue(mask_confidence, int(len(mask_confidence) * 0.4))
    #
    #     # 将original_image_mask转换为tensor，注意不修改原始numpy数组
    #     new_mask = torch.from_numpy(self.original_image_mask.copy()).to(self.data_device)
    #
    #     # 更新mask，保持类型为tensor
    #     new_mask = new_mask & (confidence >= confidence_40_percent)
    #
    #     # 将小于目标值的不确定性设置为一个极大值
    #     with torch.no_grad():
    #         new_uncertainty = self.uncertainty.clone()
    #         new_uncertainty[confidence < confidence_40_percent] = torch.finfo(torch.float32).max
    #         self.uncertainty.data = new_uncertainty
    #
    #     # 直接更新gradient_mask
    #     self.gradient_mask = new_mask
    #
    #     return confidence_40_percent
