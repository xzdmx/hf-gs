dataset=SPIn-NeRF
scene=9
depth_dir=depth_completed_lama
#depth_dir=depth_completed
inpaint_dir=lama_inpaint
#inpaint_dir=sd_inpaint

cd depth_inpainting/run
# 深度补全
echo "------------------------------------------深度补全-----------------------------------------------------"
model_path="../../checkpoints"  # absolute path

# 输入图像目录
input_rgb_dir="../../output/${dataset}/${scene}/${inpaint_dir}"
input_mask_dir="../../output/${dataset}/${scene}/mask/sub_mask"

input_depth_dir="../../output/${dataset}/${scene}/train/ours_1/depth_dis"
c2w_dir="../../output/${dataset}/${scene}/train/ours_1/c2w"
intri_dir="../../output/${dataset}/${scene}/train/ours_1/intri"

# 创建输出目录
output_base_dir="../../output/${dataset}/${scene}/${depth_dir}"

# 获取所有image_name
image_names=$(ls $input_rgb_dir | sed 's/\.png$//')

for image_name in $image_names; do
    input_rgb_path="${input_rgb_dir}/${image_name}.png"
    input_mask_path="${input_mask_dir}/${image_name}.png"
    input_depth_path="${input_depth_dir}/${image_name}.npy"
    c2w="${c2w_dir}/${image_name}.npy"
    intri="${intri_dir}/${image_name}.npy"
    output_dir="${output_base_dir}/depth_completed_${image_name}"

    # 使用 if 判断目录是否存在，如果不存在则创建
    if [! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
        echo "创建了目录: $output_dir"
    else
        echo "目录已存在: $output_dir"
    fi

    python run_inference_inpainting.py \
                --input_rgb_path $input_rgb_path \
                --input_mask $input_mask_path \
                --input_depth_path $input_depth_path \
                --model_path $model_path \
                --output_dir $output_dir \
                --denoise_steps 20 \
                --intri $intri \
                --c2w $c2w \
                --use_mask
#                --blend

#    for ((i = 1; i < 3; i++)); do
#        input_depth_path="$output_dir/${image_name}_depth_dis.npy"
#        CUDA_VISIBLE_DEVICES=0 python run_inference_inpainting.py \
#                    --input_rgb_path $input_rgb_path \
#                    --input_mask $input_mask_path \
#                    --input_depth_path $input_depth_path \
#                    --model_path $model_path \
#                    --output_dir $output_dir \
#                    --denoise_steps 20 \
#                    --intri $intri \
#                    --c2w $c2w \
#                    --use_mask
##                    --blend
#    done
done