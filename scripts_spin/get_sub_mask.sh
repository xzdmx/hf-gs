dataset=SPIn-NeRF
scene=9
last_iter=30000


start_time=$(date +%s)

cd opt_py


# 选视角
python find_camera.py --c2w_path ../output/${dataset}/${scene}/train/ours_30000/c2w/ --save_path ../output/${dataset}/${scene}/apart_45_views.txt

echo "---------------------------------------------gaussians filtering-------------------------------------------------------"
# 处理bear场景时视角要特定选取（修复视角不在生成视角内）
python gaussians_filter.py --input_ply ../output/${dataset}/${scene}/point_cloud/iteration_30000/point_cloud.ply \
                          --save_ply ../output/${dataset}/${scene}/point_cloud/iteration_1/point_cloud.ply \
                          --model_path ../output/${dataset}/${scene} \
                          --source_path ../data/${dataset}/colmap_dir/${scene} \
                          --last_iter 30000 \
                          --size 10
cd ../gaussian_splatting
python render.py -s ../data/${dataset}/colmap_dir/${scene} -m ../output/${dataset}/${scene} -u nothing --iteration 1

cd ../opt_py
# 扩大mask
python mask_edge_expend.py --input_dir ../data/${dataset}/colmap_dir/${scene}/seg \
      --output_dir ../output/${dataset}/${scene}/mask/seg_expand \
      --size 10

# mask处理
echo "---------------------------------------------mask 处理-------------------------------------------------------"
python get_inpaint_mask_2.py  --model_path ../output/${dataset}/${scene} \
      --source_path ../data/${dataset}/colmap_dir/${scene} \
      --last_iter 1 --expand_size 1


end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "执行时间：$elapsed_time 秒"