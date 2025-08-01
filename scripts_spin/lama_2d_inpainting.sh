dataset=SPIn-NeRF
scene=9

cd lama

img_path=../output/${dataset}/${scene}/train/ours_1/renders
mask_path=../output/${dataset}/${scene}/mask/sub_mask
lama_path=./LaMa_test_images/${dataset}/${scene}
key_frames_path=../output/${dataset}/${scene}/key_frames.txt

# 清除之前的输入和结果
rm -r LaMa_test_images/SPIn-NeRF/${scene}
rm -r output/SPIn-NeRF/${scene}

python prepare_lama_input.py $img_path $mask_path $lama_path $key_frames_path

export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)

out_dir=../output/${dataset}/${scene}

python bin/predict.py refine=True model.path=$(pwd)/big-lama indir=$(pwd)/LaMa_test_images/$dataset/$scene outdir=$(pwd)/output/$dataset/$scene
python prepare_pseudo_label.py $(pwd)/output/$dataset/$scene $out_dir $key_frames_path