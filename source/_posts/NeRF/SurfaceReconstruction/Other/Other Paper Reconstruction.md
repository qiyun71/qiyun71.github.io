---
title: Other Paper
date: 2023-09-21 16:00:14
tags:
  - SurfaceReconstruction
  - 3DReconstruction
  - NeRF
categories: NeRF/SurfaceReconstruction
---

Other Paper about Reconstruction

| Method | 泛化   | 数据集监督                              | 提取mesh方式                   | 获得纹理方式            |
| ------ | ------ | --------------------------------------- | ------------------------------ | ----------------------- |
| 2k2k   | 比较好 | (mesh+texture:)depth、normal、mask、rgb | 高质量深度图 --> 点云 --> mesh | 图片rgb贴图 |
| PIFu   | 比较好 | 点云(obj)、rgb(uv)、mask、camera        | 占用场 --> MC --> 点云,mesh    | 表面颜色场              |
| NeRF   | 差     | rgb、camera                             | 密度场 --> MC --> 点云,mesh    | 体积颜色场              |
| NeuS   | 差     | rgb、camera                             | SDF --> MC --> 点云,mesh       | 体积颜色场              |
| ICON   | 非常好 | rgb+mask、SMPL、法向量估计器DR          | 占用场 --> MC --> 点云,mesh    | 图片rgb贴图             |
| ECON   | 非常好 | rgb+mask、SMPL、法向量估计器DR          | d-BiNI + SC(shape completion)  | 图片rgb贴图             |

<!-- more -->

# Normal Estimation

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230930182135.png)

## ICON

> [ICON: Implicit Clothed humans Obtained from Normals](/NeRF/SurfaceReconstruction/Other/ICON)
> [ICON (mpg.de)](https://icon.is.tue.mpg.de/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230930162915.png)

## ECON

> [ECON: Explicit Clothed humans Obtained from Normals](/NeRF/SurfaceReconstruction/Other/ECON)
> [ECON: Explicit Clothed humans Optimized via Normal integration (xiuyuliang.cn)](https://xiuyuliang.cn/econ/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230930173026.png)

## 2K2K DepthEstimation

> [2K2K：High-fidelity 3D Human Digitization from Single 2K Resolution Images](/NeRF/SurfaceReconstruction/Other/2K2K)
> [High-fidelity 3D Human Digitization from Single 2K Resolution Images Project Page (sanghunhan92.github.io)](https://sanghunhan92.github.io/conference/2K2K/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230921160120.png)

## MVSNet DepthEstimation

> [MVSNet: Depth Inference for Unstructured Multi-view Stereo](/NeRF/SurfaceReconstruction/Other/MVSNet)
> [YoYo000/MVSNet: MVSNet (ECCV2018) & R-MVSNet (CVPR2019) (github.com)](https://github.com/YoYo000/MVSNet)


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231002110228.png)

# PIFu Occupancy Field

> [PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization](/NeRF/SurfaceReconstruction/Other/PIFu)
> [PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization (shunsukesaito.github.io)](https://shunsukesaito.github.io/PIFu/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230928170950.png)

## PIFuHD

> [PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization](/NeRF/SurfaceReconstruction/Other/PIFuHD)
> [PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization (shunsukesaito.github.io)](https://shunsukesaito.github.io/PIFuHD/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230928175323.png)


# Compositional

## EVA3D 2022

> 质量很低，Idea：将人体分为几个部分分别训练
> [EVA3D - Project Page (hongfz16.github.io)](https://hongfz16.github.io/projects/EVA3D.html)

![image|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230930153949.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230930154048.png)

# Loss

## Vid2Avatar

> [Vid2Avatar: 3D Avatar Reconstruction from Videos in the Wild via Self-supervised Scene Decomposition](/NeRF/SurfaceReconstruction/Other/Vid2Avatar)
> [Vid2Avatar: 3D Avatar Reconstruction from Videos in the Wild via Self-supervised Scene Decomposition (moygcc.github.io)](https://moygcc.github.io/vid2avatar/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230921171140.png)

## NISR

> [Improving Neural Indoor Surface Reconstruction with Mask-Guided Adaptive Consistency Constraints](/NeRF/SurfaceReconstruction/Other/NISR)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230925133140.png)

## D-NeuS

> [Recovering Fine Details for Neural Implicit Surface Reconstruction](/NeRF/SurfaceReconstruction/Other/D-NeuS)
> [fraunhoferhhi/D-NeuS: Recovering Fine Details for Neural Implicit Surface Reconstruction (WACV2023) (github.com)](https://github.com/fraunhoferhhi/D-NeuS)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230927202731.png)

# NeRF

## HumanRF

> [HumanRF: High-Fidelity Neural Radiance Fields for Humans in Motion (synthesiaresearch.github.io)](https://synthesiaresearch.github.io/humanrf/)
 
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231001165622.png)

## Neural Body

> [Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans (zju3dv.github.io)](https://zju3dv.github.io/neuralbody/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231001170255.png)

## InstantNVR

[Learning Neural Volumetric Representations of Dynamic Humans in Minutes (zju3dv.github.io)](https://zju3dv.github.io/instant_nvr/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231001172828.png)

## HumanGen

> [HumanGen: Generating Human Radiance Fields with Explicit Priors (suezjiang.github.io)](https://suezjiang.github.io/humangen/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231002104131.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231002104310.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231002104340.png)



# Face

## HRN

> [A Hierarchical Representation Network for Accurate and Detailed Face Reconstruction from In-The-Wild Images](/NeRF/SurfaceReconstruction/Other/HRN)
> [HRN (younglbw.github.io)](https://younglbw.github.io/HRN-homepage/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230921173632.png)

