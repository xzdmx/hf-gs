cd gaussian_splatting/submodules
## simple-knn
#pip install git+https://github.com/camenduru/simple-knn.git@60f461f4a56b7967e5d8045bf92f8c33f36976d0
## diff-gaussian-rasterization-confidence
#pip install git+https://github.com/zehaozhu/diff-gaussian-rasterization-confidence.git@1039491f922cac66331b3d638814278ae0965d3d
## pytorch3d
#pip install git+https://github.com/facebookresearch/pytorch3d.git

# or manual download and install
git clone https://github.com/zehaozhu/diff-gaussian-rasterization-confidence.git
cd diff-gaussian-rasterization-confidence
git checkout 1039491f922cac66331b3d638814278ae0965d3d
pip install .
cd ..
git clone https://github.com/camenduru/simple-knn.git
cd simple-knn
git checkout 60f461f4a56b7967e5d8045bf92f8c33f36976d0
pip install .
cd ..
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
git checkout d098beb7a7f92ee226de97b1b7905ee735aeed56
pip install .
cd ..

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

cd ..
# data example
gdown 'https://drive.google.com/uc?id=1o-YDSlHmO6NALXhmLVxB4XYkc6H78rXE'
mkdir data
unzip data.zip -d ./data

