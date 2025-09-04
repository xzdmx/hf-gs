cd gaussian_splatting/submodules
# simple-knn
pip install git+https://github.com/camenduru/simple-knn
# diff-gaussian-rasterization-confidence
pip install git+https://github.com/zehaozhu/diff-gaussian-rasterization-confidence.git
## or
#git clone https://github.com/zehaozhu/diff-gaussian-rasterization-confidence.git && cd diff-gaussian-rasterization-confidence
#pip install .
#cd ..
#git clone https://github.com/camenduru/simple-knn.git && cd simple-knn
#pip install .
#cd ..

cd ../..
# lama
git clone https://github.com/advimman/lama.git
mv prepare_lama_input.py ./lama
mv prepare_pseudo_label.py ./lama

# big lama
cd lama
curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip big-lama.zip
cd ..

# infusion checkpoint
cd checkpoint
huggingface-cli download Johanan0528/Infusion --local-dir ./

