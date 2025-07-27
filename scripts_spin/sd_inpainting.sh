dataset=SPIn-NeRF
scene=9
image_name=todo
diffuse_checkpoints_path=/media/junz/4TB-1/ldh/checkpoints/stable-diffusion-xl-1.0-inpainting-0.1

cd test_03


## 对于SPIn-NeRF这种小范围视角的数据集，直接使用给出的mask
#--input_rgb_path ../output/${dataset}/${scene}/train/ours_30000/renders/${image_name}.png
#--input_rgb_path ../output/${dataset}/${scene}/inpainted_images_dir/${image_name}.png
#--input_mask_path ../output/${dataset}/${scene}/mask/sub_mask/${image_name}.png
#python diffuse_single_image.py \
#        --input_rgb_path ../output/${dataset}/${scene}/train/ours_1/renders/${image_name}.png \
#        --input_mask_path ../output/${dataset}/${scene}/mask/sub_mask/${image_name}.png \
#        --output_dir ../output/${dataset}/${scene}/inpainted_images\
#        --diffuse_checkpoints $diffuse_checkpoints_path \
#        --prompt 'smooth stone surface texture' \
#        --negative_prompt "blurry, low quality, unnatural colors, hole, shadow, complex"

##--input_rgb_dir ../output/${dataset}/${scene}/train/ours_30000/renders  spin-nerf dataset
##--input_rgb_dir ../output/${dataset}/${scene}/train/ours_1/renders
##--input_mask_dir ../output/${dataset}/${scene}/mask/sub_mask
python diffuse_images.py \
        --input_rgb_dir ../output/${dataset}/${scene}/train/ours_30000/renders \
        --input_mask_dir ../output/${dataset}/${scene}/mask/sub_mask \
        --output_dir ../output/${dataset}/${scene}/sd_inpaint \
        --diffuse_checkpoints $diffuse_checkpoints_path \
        --model_path ../output/${dataset}/${scene} \
        --prompt 'brick wall, carpeted floor' \
        --negative_prompt "black hole, backpack, shoe"
#        blurry, low quality, unnatural colors, hole, shadow, complex, black hole
