dataset=SPIn-NeRF
scene=9
cd gaussian_splatting
# Train incomplete Gaussians
# kernel_size = 10
python train.py -s ../data/${dataset}/colmap_dir/${scene} -m ../output/${dataset}/${scene} --mask_training -u nothing --iteration 30000 --eval
#--color_aug

# Obtain c2w matrix, intrinsic matrix, incomplete depth, and rgb rendering image
python render.py -s ../data/${dataset}/colmap_dir/${scene} -m ../output/${dataset}/${scene} -u nothing --iteration 30000 --eval