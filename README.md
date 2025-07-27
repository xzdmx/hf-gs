# HF-GS

Title: High-fidelity 3D Gaussian Inpainting: preserving multi-view consistency and photorealistic details

## Installation
Install with `conda`: 
```bash
conda env create -f environment.yaml
conda activate hs-gs
```
* The contents of the `gaussian_splatting/submodules` directory are obtained by referencing  [Infusion](https://github.com/ali-vilab/Infusion)
## Download Checkpoints
Download checkpoint and put it in the 'checkpoints' folder: 
* [Infusion HuggingFace](https://huggingface.co/Johanan0528/Infusion/tree/main)
* Stable Diffusion (Choose one that you think works well. Here we provide the results of SD. The runthrough example does not need to be downloaded)

Download LaMa and big-lama. Refer to the link: [LaMa](https://github.com/advimman/lama)

**Note:** We have two python files in the lama directory used for data preparation

## Data Preparation

We have provided two scene [data](https://www.jianguoyun.com/p/DX2ZoOYQv6OSDRiW8YEGIAA) examples, packaged together, which include ground truth (GT) images, segmented masks, and viewpoint information. (The acquisition of this information refers to [Infusion](https://github.com/ali-vilab/Infusion))

The result of unzipping in the main directory of hs-gs is: 

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

## Instructions

We provide two example processes. One is the process of using Lama to repair the spinner dataset, and the other is the process of using stable diffusion to repair the MIP nerf dataset. Select one of the following to execute:

lama:

```bash
bash run_lama.sh
```

sd:

```bash
bash run_sd.sh
```

The results of the final picture display are under test or train in the scene directory `/output/[dataset]/[scene]`.

