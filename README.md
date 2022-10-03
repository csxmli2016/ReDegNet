## From Face to Natural Image: Learning Real Degradation for Blind Image Super-Resolution

[Xiaoming Li](https://csxmli2016.github.io/), [Chaofeng Chen](https://chaofengc.github.io), Xianhui Lin, [Wangmeng Zuo](http://homepage.hit.edu.cn/wangmengzuo), [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()
<a href="https://colab.research.google.com/drive/1V2M-3YLOF7bHMyZ1nfdwU61XGDepL1I9?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> 
![visitors](https://visitor-badge.glitch.me/badge?page_id=csxmli2016/ReDegNet)
[![LICENSE](https://img.shields.io/badge/LICENSE-CC%20BY--NC--SA%204.0-green)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

<p align="justify">How to design proper training pairs is critical for super-resolving real-world low-quality (LQ) images, which suffers from the difficulties in either acquiring paired ground-truth high-quality (HQ) images or synthesizing photo-realistic degraded LQ observations. Recent works mainly focus on modeling the degradation with handcrafted or estimated degradation parameters, which are however incapable to model complicated real-world degradation types, resulting in limited quality improvement. Notably, LQ face images, which may have the same degradation process as natural images (see the following Figure), can be robustly restored with photo-realistic textures by exploiting their strong structural priors. This motivates us to use the real-world LQ face images and their restored HQ counterparts to model the complex real-world degradation (namely ReDegNet), and then transfer it to HQ natural images to synthesize their realistic LQ counterparts. By taking these paired HQ-LQ face images as inputs to explicitly predict the degradation-aware and content-independent representations, we could control the degraded image generation, and subsequently transfer these degradation representations from face to natural images to synthesize the degraded LQ natural images. </p>

<img src="./GithubImgs/Fig1E8.png">

## Overview of our ReDegNet
<img src="./GithubImgs/PipelineE.png">


## Getting Start

```
git clone https://github.com/csxmli2016/ReDegNet
cd ReDegNet
conda create -n redeg python=3.8 -y
conda activate redeg
python setup.py develop
```

## Pre-train Models
Download from the following url and put them into ./experiments/weights/
- [BaiduNetDisk](https://pan.baidu.com/s/1ooHRMKSSvpFCBT9y179RSg?pwd=fr6d)
or
- [GoogleDrive](https://drive.google.com/drive/folders/1J3yUjhTrVYMMvb3O3BKqtPS6E7FotPqc?usp=sharing)


## Training
> ###### As for the real-world LQ and Pseudo HQ face pairs, you can use any robust blind face restoration methods (e.g., [GPEN](https://github.com/yangxy/GPEN), [GFPGAN](https://github.com/TencentARC/GFPGAN), [VQFR](https://github.com/TencentARC/VQFR), etc)
Prepare the training data (see examples in TrainData and options/train_redeg.yml), and
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 redeg/train.py -opt options/train_redeg.yml --launcher pytorch
```
or
```
CUDA_VISIBLE_DEVICES=0 python redeg/train.py -opt options/train_redeg.yml
```

See the training results through tensorboard
```
tensorboard --logdir tb_logger --port 7007
```

## Quick Inference of F2N-ESRGAN
```
cd CleanTest
CUDA_VISIBLE_DEVICES=0 python test_restoration.py 
```

## Synthetic Degradation Examples of ReDegNet
```
cd CleanTest
CUDA_VISIBLE_DEVICES=0 python test_degradation.py 
```

## Acknowledgement
This project is built based on the excellent [BasicSR](https://github.com/XPixelGroup/BasicSR) and [KAIR](https://github.com/cszn/KAIR).


## Citation

```
@InProceedings{Li_2022_ReDegNet,
author = {Li, Xiaoming and Chen, Chaofeng and Lin, Xianhui and Zuo, Wangmeng and Zhang, Lei},
title = {From Face to Natural Image: Learning Real Degradation for Blind Image Super-Resolution},
booktitle = {ECCV},
year = {2022}
}
```

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

