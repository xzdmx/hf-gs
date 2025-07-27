bash scripts_spin/mask_train.sh
bash scripts_spin/get_sub_mask.sh

# Select two perspectives from apart_45_views.txt and save them as key_frames.txt
# (you can also directly select the better result perspective as key_frame by inpaint from the perspective of apart_45_views.txt).
# The default perspective is used here
cp ./data/SPIn-NeRF/colmap_dir/9/key_frames.txt ./output/SPIn-NeRF/9

bash scripts_spin/lama_2d_inpainting.sh
bash scripts_spin/depth_inpainting.sh
bash scripts_spin/3d_inpainting.sh