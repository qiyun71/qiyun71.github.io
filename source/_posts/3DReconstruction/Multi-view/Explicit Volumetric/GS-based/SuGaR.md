---
title: SuGaR
date: 2023-12-19 14:26:13
tags: 
categories: 3DReconstruction/Multi-view/Explicit Volumetric/GS-based
---

| Title     | SuGaR                                                                                                                                                                                                   |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | `Gu{\'e}don, Antoine and Lepetit, Vincent`                                                                                                                                                              |
| Conf/Jour | CVPR                                                                                                                                                                                                    |
| Year      | 2024                                                                                                                                                                                                    |
| Project   | [Anttwo/SuGaR: Official implementation of SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering (github.com)](https://github.com/Anttwo/SuGaR) |
| Paper     | [SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering](https://arxiv.org/pdf/2311.12775)                                                      |

<!-- more -->





# 实验(win10)

~~conda env create -f environment.yml~~
复制environment.yml中pip包到requirements.txt中，通过pip安装
`conda create -n sugar`
`pip install -r requirements.txt`
add:
- plyfile
- tqdm
- rich

`pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118`

pytorch3d: [pytorch3D Windows下安装经验总结_windows安装pytorch3d-CSDN博客](https://blog.csdn.net/m0_70229101/article/details/127196699)


```bash
cd gaussian_splatting/submodules/diff-gaussian-rasterization/
pip install -e .
cd ../simple-knn/
pip install -e .
cd ../../../
```

## run

```bash
# GS
python gaussian_splatting/train.py -s <path to COLMAP or NeRF Synthetic dataset> --iterations 7000 -m <path to the desired output directory>

-

## eg
python gaussian_splatting/train.py -s inputs/dtu_scan114 --iterations 7000 -m exp/dtu_scan114
python gaussian_splatting/train.py -s inputs/Miku --iterations 7000 -m exp/Miku

# SuGaR
python train.py -s <path to COLMAP or NeRF Synthetic dataset> -c <path to the Gaussian Splatting checkpoint> -r <"density" or "sdf">

##eg
python train.py -s inputs/dtu_scan114 -c exp/dtu_scan114/ -r sdf
python train.py -s inputs/Miku -c exp/Miku/ -r sdf
```