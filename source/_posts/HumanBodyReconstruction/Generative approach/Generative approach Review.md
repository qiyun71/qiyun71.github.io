---
title: Generative approach Review
date: 2023-10-24 11:56:04
tags:
  - Diffusion
  - GAN
  - Flow
  - Review
categories: HumanBodyReconstruction/Generative approach
top: true
---

| Paper     | Model                  | Input        | Parameter/Pnum | GPU         |
| --------- | ---------------------- | ------------ | -------------- | ----------- |
| DiT-3D    | Diffusion Transformers | Voxelized PC |                |             |
| PointFlow | AE flow-based          | PointCloud   | 1.61M          |             |
| FlowGAN   | GAN flow-based         | Single Image | N = 2500       | A40 45GB    |
| BuilDiff  | Diffusion models       | Single Image | 1024 to 4096   | A40 45GB    |
| CCD-3DR   | CDPM                   | Single Image | 8192           | 3090Ti 24GB |
| SG-GAN    | SG-GAN                 | Single Image |                |             |
|           |                        |              |                |             |

[GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images (nv-tlabs.github.io)](https://nv-tlabs.github.io/GET3D/)
[NVlabs/nvdiffrec: Official code for the CVPR 2022 (oral) paper "Extracting Triangular 3D Models, Materials, and Lighting From Images". (github.com)](https://github.com/NVlabs/nvdiffrec)
[pix2pix3D: 3D-aware Conditional Image Synthesis (cmu.edu)](http://www.cs.cmu.edu/~pix2pix3D/)

<!-- more -->

# Generative approach

## Network Framework

![Image.png|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021163349.png)

- GAN(generative adversarial networks)
- VAE(variational auto-encoders)
- auto-regressive models
- Normalized flows(flow-based models), PointFlow
  - 相当于多个生成器，并且可逆
  - [Flow-based Generative Model - YouTube](https://www.youtube.com/watch?v=uXY18nzdSsM&list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4&index=60)
  - [Flow-based Generative Model 笔记整理 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/267305869)
  - [Normalization Flow (标准化流) 总结 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/59615785)
- DiT(Diffusion Transformers), DiT-3D

Flow-based：
![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021100952.png)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021101055.png)

## Loss

点云倒角距离 CD ↓
$\begin{aligned}\mathcal{L}_{CD}&=\sum_{y'\in Y'}min_{y\in Y}||y'-y||_2^2+\sum_{y\in Y}min_{y'\in Y'}||y-y'||_2^2,\end{aligned}$

推土距离 EMD (Earth Mover's distance)↓
$\mathcal{L}_{EMD}=min_{\phi:Y\rightarrow Y^{\prime}}\sum_{x\in Y}||x-\phi(x)||_{2}$ , φ indicates a parameter of bijection.


## Diffusion Models

### BuilDiff

预计 2023.11 release
[BuilDiff论文阅读笔记](/HumanBodyReconstruction/Generative%20approach/BuilDiff)
[weiyao1996/BuilDiff: BuilDiff: 3D Building Shape Generation using Single-Image Conditional Point Cloud Diffusion Models (github.com)](https://github.com/weiyao1996/BuilDiff)
[BuilDiff: 3D Building Shape Generation using Single-Image Conditional Point Cloud Diffusion Models (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4796303379219349505&noteId=2014132586369911808)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021114740.png)

### RODIN

[RODIN Diffusion (microsoft.com)](https://3d-avatar-diffusion.microsoft.com/)
[Rodin: A Generative Model for Sculpting 3D Digital Avatars Using Diffusion (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4700249091733454849&noteId=2024885528733762560)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231028214552.png)

### CCD-3DR

[CCD-3DR: Consistent Conditioning in Diffusion for Single-Image 3D Reconstruction (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4789424767274844161&noteId=2014146864821066240)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021115606.png)

### PC^2

[PC2: Projection-Conditioned Point Cloud Diffusion for Single-Image 3D Reconstruction (lukemelas.github.io)](https://lukemelas.github.io/projection-conditioned-point-cloud-diffusion/)
[$PC^2$: Projection-Conditioned Point Cloud Diffusion for Single-Image 3D Reconstruction (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4726675454702911489&noteId=2014671879272477696)

相机位姿???
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021203845.png)

### DiT-3D

[DiT-3D论文阅读笔记](/HumanBodyReconstruction/Generative%20approach/DiT-3D)
[DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation](https://dit-3d.github.io/)
[DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4776143720479195137&noteId=2011558450133224704)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231019170328.png)

### Etc

Colored PC [3D Colored Shape Reconstruction from a Single RGB Image through Diffusion (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4723055356805120001&noteId=2014681937165010176)

## GAN

### FlowGAN

[FlowGAN论文阅读笔记](/HumanBodyReconstruction/Generative%20approach/FlowGAN)
[Flow-based GAN for 3D Point Cloud Generation from a Single Image (mpg.de)](https://bmvc2022.mpi-inf.mpg.de/569/)
[Flow-based GAN for 3D Point Cloud Generation from a Single Image (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4677468038203719681&noteId=2011515461854903552)

![image.png|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231019162233.png)

### SG-GAN

[SG-GAN论文阅读笔记](/HumanBodyReconstruction/Generative%20approach/SG-GAN)
[SG-GAN: Fine Stereoscopic-Aware Generation for 3D Brain Point Cloud Up-sampling from a Single Image (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4762223320204574721&noteId=2014610279544785920)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021193539.png)

### 3D Brain Reconstruction and Complete

[3D Brain Reconstruction by Hierarchical Shape-Perception Network from a Single Incomplete Image. (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4552080679232348161&noteId=2015496104888593408)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022101838.png)

## NFs (Normalizing Flows)

### PointFlow

[stevenygd/PointFlow: PointFlow : 3D Point Cloud Generation with Continuous Normalizing Flows (github.com)](https://github.com/stevenygd/PointFlow?tab=readme-ov-file)
[PointFlow: 3D Point Cloud Generation with Continuous Normalizing Flows (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4500221890681004033&noteId=2011462053836873984)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231019154043.png)

## Other

### Automatic Reverse Engineering

[Automatic Reverse Engineering: Creating computer-aided design (CAD) models from multi-view images (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4804297472033685505&noteId=2014396749893884416)

多视图图像生成 CAD 命令序列
局限性：

- CAD 序列的长度仍然局限于 60 个命令，因此只支持相对简单的对象
- 表示仅限于平面和圆柱表面，而许多现实世界的对象可能包括更灵活的三角形网格或样条表示

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021160431.png)

### SimIPU

[zhyever/SimIPU: [AAAI 2021] Official Implementation of "SimIPU: Simple 2D Image and 3D Point Cloud Unsupervised Pre-Training for Spatial-Aware Visual Representations" (github. Com)]( https://github.com/zhyever/SimIPU )

雷达点云+图片
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022103332.png)

### 3DRIMR

[3DRIMR: 3D Reconstruction and Imaging via mmWave Radar based on Deep Learning. (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4542232848617857025&noteId=2015491146701999104)
MmWave Radar + GAN
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022102437.png)

### TransHuman

[TransHuman论文阅读笔记](TransHuman.md)
[TransHuman: A Transformer-based Human Representation for Generalizable Neural Human Rendering (pansanity666.github.io)](https://pansanity666.github.io/TransHuman/)

ImplicitFunction(NeRF)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022103225.png)

### SE-MD

[SE-MD: A Single-encoder multiple-decoder deep network for point cloud generation from 2D images. (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4546363961368010753&noteId=2015501024673795072)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022105308.png)

### Fusion of cross-view images

[3D Reconstruction through Fusion of Cross-View Images (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4546340788354310145&noteId=2015534437671973888)

多张卫星图片配准
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022110219.png)

### GET3D

[nv-tlabs/GET3D (github.com)](https://github.com/nv-tlabs/GET3D)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023154445.png)


### nvdiffrec

[NVlabs/nvdiffrec: Official code for the CVPR 2022 (oral) paper "Extracting Triangular 3D Models, Materials, and Lighting From Images". (github.com)](https://github.com/NVlabs/nvdiffrec)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023154513.png)

### Make-It-3D

[Make-It-3D: High-Fidelity 3D Creation from A Single Image with Diffusion Prior](https://make-it-3d.github.io/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023154817.png)

### HyperHuman

高质量人类图片
[HyperHuman: Hyper-Realistic Human Generation with Latent Structural Diffusion (snap-research.github.io)](https://snap-research.github.io/HyperHuman/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023155025.png)

### AG3D

[AG3D: Learning to Generate 3D Avatars from 2D Image Collections (zj-dong.github.io)](https://zj-dong.github.io/AG3D/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023160511.png)


### SyncDreamer

[SyncDreamer: Generating Multiview-consistent Images from a Single-view Image (liuyuan-pal.github.io)](https://liuyuan-pal.github.io/SyncDreamer/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023165400.png)

### One-2-3-45

[One-2-3-45](https://one-2-3-45.github.io/)

MVS+NeRF
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023165804.png)

### Zero-1-to-3

[Zero-1-to-3: Zero-shot One Image to 3D Object (columbia.edu)](https://zero123.cs.columbia.edu/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023171814.png)

### DreamGaussian

[DreamGaussian](https://dreamgaussian.github.io/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231024164334.png)

### Wonder3D

[Wonder3D: Single Image to 3D using Cross-Domain Diffusion (xxlong.site)](https://www.xxlong.site/Wonder3D/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231024205005.png)

#### CLIP

[openai/CLIP: CLIP (Contrastive Language-Image Pretraining), Predict the most relevant text snippet given an image (github.com)](https://github.com/openai/CLIP)

对比语言-图片预训练模型
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231025093839.png)

