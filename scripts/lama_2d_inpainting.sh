dataset=SPIn-NeRF
scene=3

cd lama

img_path=../output/${dataset}/${scene}/train/ours_30000/renders
mask_path=../output/${dataset}/${scene}/mask/sub_mask
lama_path=./LaMa_test_images/${dataset}/${scene}
key_frames_path=../output/${dataset}/${scene}/key_frames.txt

python prepare_lama_input.py $img_path $mask_path $lama_path $key_frames_path

export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)

out_dir=../output/${dataset}/${scene}

python bin/predict.py refine=True model.path=$(pwd)/big-lama indir=$(pwd)/LaMa_test_images/$dataset/$scene outdir=$(pwd)/output/$dataset/$scene
python prepare_pseudo_label.py $(pwd)/output/$dataset/$scene $out_dir $key_frames_path