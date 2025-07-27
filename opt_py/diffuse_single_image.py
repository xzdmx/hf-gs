import argparse
import os

import numpy as np
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_rgb_path', required=True, help='Path to the input RGB image.')
    parser.add_argument('--input_mask_path', required=True, help='Path to the input mask image.')
    parser.add_argument('--output_dir', required=True, help='Path to save the output diffused image.')
    parser.add_argument('--diffuse_checkpoints', required=True, help='Path to the diffusion checkpoints path.')
    parser.add_argument("--prompt", type=str,default='',help="diffuse prompt")
    parser.add_argument("--negative_prompt", type=str,default='',help="diffuse negative_prompt")

    args = parser.parse_args()

    pipe = AutoPipelineForInpainting.from_pretrained(
        args.diffuse_checkpoints,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    print(f'----------------------------diffuse single image---------------------------------')
    print(f'Input rgb path: {args.input_rgb_path}')
    print(f'Input mask path: {args.input_mask_path}')
    print(f'Output directory: {args.output_dir}')

    img_url = args.input_rgb_path
    mask_url = args.input_mask_path
    output_dir = args.output_dir
    # 如果输出目录不存在则创建目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    prompt = args.prompt
    negative_prompt = args.negative_prompt
    # 随机生成seed
    seed = np.random.randint(0, 2**32 - 1)
    # seed = 2472008763
    print("seed is {}".format(seed))

    image = load_image(img_url)
    w, h = image.size
    image = image.resize((1024, 1024))
    mask_image = load_image(mask_url).resize((1024, 1024))

    generator = torch.Generator(device="cuda").manual_seed(seed)  # 每次循环中重新初始化生成器
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=7.5,
        num_inference_steps=40,
        strength=0.99,
        generator=generator,
    ).images[0]
    image = image.resize((w, h))
    # 保存
    image_name_with_extension = os.path.basename(img_url)
    save_path = os.path.join(output_dir, image_name_with_extension)
    image.save(save_path)
