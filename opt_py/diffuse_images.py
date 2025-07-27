import argparse
import os

import cv2
import numpy as np
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch


def get_file_suffix(input_dir):
    """
    从输入目录中获取一个文件，并返回其后缀名。
    如果目录为空，则返回None。
    """
    file_list = os.listdir(input_dir)
    if file_list:
        first_file = file_list[0]
        _, file_suffix = os.path.splitext(first_file)
        return file_suffix
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_rgb_dir', required=True, help='Path to the input RGB image directory.')
    parser.add_argument('--input_mask_dir', required=True, help='Path to the input mask image directory.')
    parser.add_argument('--output_dir', required=True, help='Path to save the output diffused image directory.')
    parser.add_argument('--diffuse_checkpoints', required=True, help='Path to the diffusion checkpoints path.')
    parser.add_argument('--model_path', required=True, help='Path to the scene path.')
    parser.add_argument("--prompt", type=str, default='', help="diffuse prompt")

    args = parser.parse_args()

    pipe = AutoPipelineForInpainting.from_pretrained(
        args.diffuse_checkpoints,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    # 相隔45°的视角
    frames = []
    with open(f"{args.model_path}/apart_45_views.txt", 'r') as f:
        for line in f:
            frames.append(line.strip())

    mask_images = os.listdir(args.input_mask_dir)
    prompt = args.prompt
    # 随机生成seed
    seed = np.random.randint(0, 10000)

    # 获取RGB图像目录下文件的后缀，用于后续构建完整文件名
    rgb_file_suffix = get_file_suffix(args.input_rgb_dir)
    if not rgb_file_suffix:
        print("输入的RGB图像目录为空，请检查目录内容。")
        exit(1)

    for mask_name in mask_images:
        base_name = os.path.splitext(mask_name)[0]
        if base_name in frames:
            # 构建原始RGB图像的完整文件名
            rgb_full_name = base_name + rgb_file_suffix
            img_url = os.path.join(args.input_rgb_dir, rgb_full_name)
            mask_url = os.path.join(args.input_mask_dir, mask_name)

            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            image = load_image(img_url)
            w, h = image.size
            image = image.resize((1024, 1024))
            mask_image = load_image(mask_url).resize((1024, 1024))

            generator = torch.Generator(device="cuda").manual_seed(seed)
            image = pipe(
                prompt=prompt,
                image=image,
                mask_image=mask_image,
                guidance_scale=9.0,
                num_inference_steps=40,
                strength=0.99,
                generator=generator,
            ).images[0]
            image = image.resize((w, h))
            # 保存
            save_name = base_name + '.png'
            save_path = os.path.join(args.output_dir, save_name)
            image.save(save_path)