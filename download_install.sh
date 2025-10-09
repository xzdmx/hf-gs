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

