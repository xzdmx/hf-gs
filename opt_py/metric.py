import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import lpips
from pytorch_fid import fid_score
import pandas as pd


def main():
    # 定义目录和文件路径
    result_dir = '/media/junz/4TB-1/ldh/papers/0_Infusion/output/SPIn-NeRF/3/test/ours_1504/renders'
    gt_dir = '/media/junz/4TB-1/ldh/papers/0_Infusion/output/SPIn-NeRF/3/test/ours_1504/gt'
    csv_path = '/media/junz/4TB-1/ldh/papers/0_Infusion/output/SPIn-NeRF/3/score_1504.csv'
    # result_dir = '/media/junz/4TB-1/ldh/papers/GScream/outputs/spinnerf_dataset/1/gscream/test/ours_30000/renders'
    # gt_dir = '/media/junz/4TB-1/ldh/papers/GScream/outputs/spinnerf_dataset/1/gscream/test/ours_30000/gt'
    # csv_path = '/media/junz/4TB-1/ldh/papers/GScream/outputs/spinnerf_dataset/1/score.csv'

    # 确保结果目录和 GT 目录存在
    if not os.path.exists(result_dir):
        raise FileNotFoundError(f"结果目录 {result_dir} 不存在。")
    if not os.path.exists(gt_dir):
        raise FileNotFoundError(f"GT 目录 {gt_dir} 不存在。")

    # 存储结果的列表
    image_names = []
    lpips_losses = []

    # 加载预训练的 LPIPS 模型
    lpips_model = lpips.LPIPS(net='alex', version=0.1).cuda()
    # 加载预训练的 Inception-v3 模型
    inception_model = models.inception_v3(pretrained=True).cuda()
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 获取结果目录和 GT 目录中的文件列表
    result_files = sorted(os.listdir(result_dir))
    gt_files = sorted(os.listdir(gt_dir))

    # 确保结果目录和 GT 目录中的文件数量相同
    if len(result_files)!= len(gt_files):
        raise ValueError("结果目录和 GT 目录中的文件数量不相等。")

    for result_file, gt_file in zip(result_files, gt_files):
        # 构建完整的文件路径
        result_path = os.path.join(result_dir, result_file)
        gt_path = os.path.join(gt_dir, gt_file)

        # Load images
        img0 = lpips.im2tensor(lpips.load_image(result_path)).cuda()  # RGB image from [-1,1]
        img1 = lpips.im2tensor(lpips.load_image(gt_path)).cuda()

        # 计算 LPIPS 损失
        lpips_loss = lpips_model.forward(img0, img1).item()
        lpips_losses.append(lpips_loss)

        image_names.append(result_file)

    # 计算平均 LPIPS 损失
    avg_lpips_loss = sum(lpips_losses) / len(lpips_losses)

    # 计算 FID 损失
    avg_fid_loss = fid_score.calculate_fid_given_paths([gt_dir, result_dir], batch_size=32, device='cuda:0', dims=2048)

    # 存储结果到 DataFrame
    data = {
        'Image Name': image_names,
        'LPIPS Loss': lpips_losses,
        'FID Loss': [None] * len(lpips_losses)  # 为每个图像的 FID 损失填充 None
    }
    df = pd.DataFrame(data)
    # 计算平均 LPIPS 和 FID 损失
    df.loc[len(df)] = ['Average', avg_lpips_loss, avg_fid_loss]

    # 保存 DataFrame 到 CSV 文件
    df.to_csv(csv_path, index=False)
    print(f"指标已保存到 {csv_path}")


if __name__ == "__main__":
    main()
