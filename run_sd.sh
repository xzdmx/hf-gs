# garden
bash scripts/mask_train.sh
bash scripts/get_sub_mask.sh
# use the sd results provided
cp -r ./data/Mip-NeRF/colmap_dir/garden/sd_inapint ./output/Mip-NeRF/garden

# Select two perspectives from apart_45_views.txt and save them as key_frames.txt
# (you can also directly select the better result perspective as key_frame by inpaint from the perspective of apart_45_views.txt).
# The default perspective is used here
cp ./data/Mip-NeRF/colmap_dir/garden/key_frames.txt ./output/Mip-NeRF/garden

bash scripts/depth_inpainting.sh
bash scripts/3d_inpainting.sh