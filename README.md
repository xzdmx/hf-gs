# HF-GS

Title: High-fidelity 3D Gaussian Inpainting: preserving multi-view consistency and photorealistic details.

![Pipeline](assets/pipeline.jpg)

## Environment 
* GPU Requirement: NVIDIA RTX 3090 (24GB VRAM)
* Operating System: Ubuntu 20.04 LTS
## Installation
Install with `conda`: 
```bash
conda env create -f environment.yaml
conda activate hf-gs
```

## Download Checkpoints & Data Preparation
**You can simply run the command below to download and install checkpoints and data. If it fails, you can refer to the other download methods provided in this file.**
```bash
bash download_install.sh
```
It include checkpoints as followed: 
* [Infusion Model Checkpoint](https://huggingface.co/Johanan0528/Infusion/tree/main): Consists of multiple files (not a single file), with a total size of **6.86GB**. It includes pre-trained weights, optimizer states, and training configuration details (for direct loading in inference or further fine-tuning).
* Stable Diffusion (Choose one that you think works well. Here we provide the results of SD. The runthrough example does not need to be downloaded)
*  `big-lama`: Compressed file size of **364MB**, used specifically for image inpainting tasks (to repair or fill in missing/defective regions of images), Refer to the link: [LaMa](https://github.com/advimman/lama).

**Note:** We have two python files in the lama directory used for data preparation

We have provided two scene [data](https://drive.google.com/drive/folders/1aUuvNQZvUwt93CfFBg_ZT2E8Uz_AfC9h?usp=drive_link) examples (packaged together, with a total size of **147MB**), which include ground truth (GT) images, segmented masks, and viewpoint information. (The acquisition of this information refers to [Infusion](https://github.com/ali-vilab/Infusion)).

You have downloaded it by running `download_install.sh`. The result of unzipping in the main directory of hs-gs is: 

```text
data
├── Mip-NeRF
│   └── colmap_dir
│       └── garden
│           ├── images
│           ├── seg
│           └── sparse
│               └── 0
└── SPIn-NeRF
    └── colmap_dir
        └── 9
            ├── images
            ├── seg
            └── sparse
                └── 0
```

- **Note:** Due to the uncertainty of stable diffusion, we provide the repair results for the example.

* If gdown download fails, you can click this [link](https://drive.google.com/drive/folders/1aUuvNQZvUwt93CfFBg_ZT2E8Uz_AfC9h?usp=drive_link) to download it manually. (or download the [data.zip](https://drive.google.com/file/d/1o-YDSlHmO6NALXhmLVxB4XYkc6H78rXE/view?usp=drive_link))


## Instructions

We provide two example processes. One is the process of using Lama to repair the spinnerf dataset, and the other is the process of using stable diffusion to repair the Mip-NeRF dataset. Select one of the following to execute:

lama:

```bash
bash run_lama.sh
```

sd:

```bash
bash run_sd.sh  # todo
```
* The running time is about **20min**.

* The results of the final picture display are under test or train in the scene directory `/output/[dataset]/[scene]`.

