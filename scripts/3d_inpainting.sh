main_image="DSC08105" # 主视角
dataset=Mip-NeRF
scene=garden
image_name=DSC08105
last_iter=1
iter=10000
#depth_dir=depth_completed_lama
depth_dir=depth_completed
#inpaint_dir=../output/${dataset}/${scene}/lama_inpaint
inpaint_dir=../output/${dataset}/${scene}/sd_inpaint

origin_ply="./output/${dataset}/${scene}/point_cloud/iteration_${last_iter}/point_cloud.ply"
supp_ply="./output/${dataset}/${scene}/depth_completed/depth_completed_${image_name}/${image_name}_mask.ply"
save_ply="./output/${dataset}/${scene}/point_cloud/iteration_30001/point_cloud.ply"
# Combine inpainted Gaussians and incomplete Gaussians.
python compose.py --original_ply $origin_ply  --supp_ply $supp_ply --save_ply $save_ply --nb_points 100 --threshold 1.0

cd gaussian_splatting

python train1.py -s ../data/${dataset}/colmap_dir/${scene} -m ../output/${dataset}/${scene} --load_iteration 30001 \
        --iteration $iter --inpaint_dir $inpaint_dir --main_image $main_image -u nothing --port 6010 \
        --depth_dir $depth_dir

### Render
python render.py -s ../data/${dataset}/colmap_dir/${scene} -m ../output/${dataset}/${scene} -u nothing --iteration $iter