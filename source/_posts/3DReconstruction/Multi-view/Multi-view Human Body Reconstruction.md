---
title: Multi-view Human Body Reconstruction
date: 2023-10-09T16:33:31.000Z
tags:
  - ClothedHumans
  - 3DReconstruction
  - PointCloud
categories: 3DReconstruction/Multi-view
date updated: 2023-11-05T16:50:36.000Z
---

![Human.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/Human.png)

Terminology/Jargon

- Human Radiance Fields
- 3D **Clothed Human** Reconstruction | **Digitization**

Application

- 三维重建设备：手持扫描仪或 360 度相机矩阵（成本高）
- [复刻一个迷你版的自己](https://www.yangtse.com/content/1604507html)

Method

1. **Depth&Normal Estimation**(2K2K) 
2. **Implicit Function**(PIFu or NeRF) 
3. **Generative approach**  [Generative Models Reconstruction](Generative%20Models%20Reconstruction.md)


**Awesome Human Body Reconstruction**

| Method | 泛化  | 数据集监督                                | 提取 mesh 方式                    | 获得纹理方式    |
| ------ | --- | ------------------------------------ | ----------------------------- | --------- |
| 2k2k   | 比较好 | (mesh+texture:)depth、normal、mask、rgb | 高质量深度图 --> 点云 --> mesh        | 图片 rgb 贴图 |
| PIFu   | 比较好 | 点云(obj)、rgb(uv)、mask、camera          | 占用场 --> MC --> 点云,mesh        | 表面颜色场     |
| NeRF   | 差   | rgb、camera                           | 密度场 --> MC --> 点云,mesh        | 体积颜色场     |
| NeuS   | 差   | rgb、camera                           | SDF --> MC --> 点云,mesh        | 体积颜色场     |
| ICON   | 非常好 | rgb+mask、SMPL、法向量估计器 DR              | 占用场 --> MC --> 点云,mesh        | 图片 rgb 贴图 |
| ECON   | 非常好 | rgb+mask、SMPL、法向量估计器 DR              | d-BiNI + SC(shape completion) | 图片 rgb 贴图 |

<!-- more -->

# 人体三维重建方法综述

## Implicit Function

**方法 0**：训练隐式函数表示
(eg: NeRF、PIFu、ICON)
**DoubleField**(多视图)

***问题：需要估计相机位姿，估计方法有一定的误差，视图少时误差更大***

## Depth&Normal Estimation

**方法 1**：深度估计+多视图深度图融合 or 多视图点云配准
(2K2K-based)

深度估计: 2K2K、MVSNet、ECON...

- 多视图深度图融合：[DepthFusion: Fuse multiple depth frames into a point cloud](https://github.com/touristCheng/DepthFusion)
  - 需要相机位姿，位姿估计有误差
  - 更准确的位姿: BA(Bundle Adjusted 光束法平差，优化相机 pose 和 landmark)

- 多视图点云配准：[Point Cloud Registration](PointCloud%20Review.md)
  - **点云配准**(Point Cloud Registration) 2K 生成的多角度点云形状不统一

***问题：无法保证生成的多视角深度图具有多视图一致性***

## Generative approach

**方法 2**：生成式方法由图片生成点云
Generative approach(Multi-view image、pose (keypoints)... --> PointCloud)
1. 扩散模型
  1. 直接生成点云 *BuilDiff*
  2. 生成三平面特征+NeRF *RODIN*
  3. 多视图 Diffusion [DiffuStereo](https://liuyebin.com/diffustereo/diffustereo.html)
2. GAN 网络生成点云 *SG-GAN*
3. 生成一致性图片+NeRF

- 参考 [BuilDiff](https://github.com/weiyao1996/BuilDiff)，构建网络([PVCNNs](https://readpaper.com/pdf-annotate/note?pdfId=4544669809538392065&noteId=2018413897297176576) 单类训练)
  - 是否更换扩散网络 [DiT-3D](https://dit-3d.github.io/)，可以学习显式的类条件嵌入(生成多样化的点云)
  - 是否依靠 SMPL，根据 LBS(Linear Blending Skinning)将人体 mesh 变形到规范化空间
    - [Video2Avatar](https://moygcc.github.io/vid2avatar/) (NeRF-based)将整个人体规范化后采样
    - [EVA3D](https://hongfz16.github.io/projects/EVA3D) 将 NeRF 融入 GAN 生成图片，并与真实图片一同训练判别器(人体规范化后分块 NeRF)

***问题：直接生成点云或者对点云进行扩散优化，会花费大量的内存***

## 混合方法

**方法 3**：组合深度估计 + 生成式方法（缝合多个方法）
[HaP](https://github.com/yztang4/HaP)：深度估计+SMPL 估计+Diffusion Model 精细化

***问题：依赖深度估计和 SMPL 估计得到的结果***

**方法 4**：隐函数 + 生成式方法 + 非刚ICP配准
[DiffuStereo](https://liuyebin.com/diffustereo/diffustereo.html)：NeRF(DoubleField) + Diffusion Model + non-rigid ICP （***不开源***）

# 三维重建方法流程对比

## Implicit Function

### NeRF

![NeuS2.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231024153406.png)
预测每个采样点 sdf 和 feature 向量
$(sdf,\mathbf{feature})=f_\Theta(\mathbf{e}),\quad\mathbf{e}=(\mathbf{x},h_\Omega(\mathbf{x})).$

预测每个采样点颜色值
$\mathbf c=c_{\Upsilon}(\mathbf x,\mathbf n,\mathbf v,sdf,\mathbf{feature})$，$\mathbf n=\nabla_\mathbf x sdf.$

体渲染像素颜色
$\hat{C}=\sum_{i=1}^n T_i\alpha_i c_i$， $T_i=\prod_{j=1}^{i-1}(1-\alpha_j)$ ，$\alpha_i=\max\left(\frac{\Phi_s(f(\mathbf{p}(t_i))))-\Phi_s(f(\mathbf{p}(t_{i+1})))}{\Phi_s(f(\mathbf{p}(t_i)))},0\right)$

训练得到 MLP，根据 MarchingCube 得到点云

### PIFu

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230928170950.png)

将输入图像中每个像素的特征通过 MLP 映射为占用场

## Depth&Normal Estimation

![2K2K.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230921160120.png)

预测低分辨率法向量图和深度图，$\hat M$ 为预测出的 mask
$\mathbf{D}^l=\hat{\mathbf{D}}^l\odot\hat{\mathbf{M}}^l$， $\hat{\mathbf{D}}^l,\hat{\mathbf{M}}^l,\mathbf{N}^l=G^l_{\mathbf{D}}(I^l)$

预测高分辨率 part 法向量图，M 为变换矩阵
$\bar{\mathbf{n}}_i=G_{\mathbf{N},i}(\bar{\mathbf{p}}_i,\mathbf{M}_i^{-1}\mathbf{N}^l)$， $\bar{\mathbf{p}}_i=\mathbf{M}_i\mathbf{p}_i,$

拼接为高分辨率整体法向量图
$\mathbf{N}^h=\sum\limits_{i=1}^K\left(\mathbf{W}_i\odot\mathbf{n}_i\right)$ ，$\mathbf{n}_i=\mathbf{M}_i^{-1}\bar{\mathbf{n}}_i$

预测高分辨率深度图
$\mathbf{D}^h=\hat{\mathbf{D}}^h\odot\hat{\mathbf{M}}^h$，$\hat{\mathbf{D}}^h,\hat{\mathbf{M}}^h=G^h_{\mathbf{D}}(\mathbf{N}^h,\mathbf{D}^l)$

深度图转点云

## Generative approach

### Diffusion Model Network

[Diffusion Model Network学习笔记](Diffusion%20Models.md)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021114740.png)

![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231024111221.png)

**3D CNN**: PVCNN、PointNet、PointNet++

**2D CNN:** 3D-aware convolution(RODIN)

### GAN

---

# Paper about Human Reconstruction👇

# NeRF-based Human Body Reconstruction

## HISR

[[2312.17192] HISR: Hybrid Implicit Surface Representation for Photorealistic 3D Human Reconstruction (arxiv.org)](https://arxiv.org/abs/2312.17192)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240104172512.png)

- 对不透明区域（例如身体、脸部、衣服）执行基于表面的渲染
- 在半透明区域（例如头发）上执行体积渲染

## DoubleField

[DoubleField Project Page (liuyebin.com)](http://www.liuyebin.com/dbfield/dbfield.html)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231110163602.png)

## Learning Visibility Field for Detailed 3D Human Reconstruction and Relighting

[Learning Visibility Field for Detailed 3D Human Reconstruction and Relighting (thecvf.com)](https://openaccess.thecvf.com/content/CVPR2023/papers/Zheng_Learning_Visibility_Field_for_Detailed_3D_Human_Reconstruction_and_Relighting_CVPR_2023_paper.pdf)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008104907.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008105237.png)

## HumanGen

> [HumanGen: Generating Human Radiance Fields with Explicit Priors (suezjiang.github.io)](https://suezjiang.github.io/humangen/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231002104131.png)

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231002104310.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231002104340.png)

## GNeuVox

[GNeuVox: Generalizable Neural Voxels for Fast Human Radiance Fields (taoranyi.com)](https://taoranyi.com/gneuvox/)
[Generalizable Neural Voxels for Fast Human Radiance Fields (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4738288024060706817&noteId=1996978666924478208)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008173458.png)

## CAR

[CAR (tingtingliao.github.io)](https://tingtingliao.github.io/CAR/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008172759.png)

## HDHumans

[HDHumans (acm.org)](https://dl.acm.org/doi/pdf/10.1145/3606927)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008193531.png)

## EVA3D 2022

Compositional Human body
质量很低
Idea：

- 将人体分为几个部分分别训练
- 将 NeRF 融合进 GAN 的生成器中，并与一个判别器进行联合训练

Cost：

- 8 NVIDIA V100 Gpus for 5 days

> [EVA3D - Project Page (hongfz16.github.io)](https://hongfz16.github.io/projects/EVA3D.html)
> [EVA3D: Compositional 3D Human Generation from 2D Image Collections (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4677480793493209089&noteId=1985412009585125888)

![image|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230930153949.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230930154048.png)

## Dynamic

### 3DGS-Avatar

[3DGS-Avatar: Animatable Avatars via Deformable 3D Gaussian Splatting (neuralbodies.github.io)](https://neuralbodies.github.io/3DGS-Avatar/)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231218201652.png)


### GaussianAvatar

[Projectpage of GaussianAvatar (huliangxiao.github.io)](https://huliangxiao.github.io/GaussianAvatar)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231205153138.png)

### Vid2Avatar

> [Vid2Avatar: 3D Avatar Reconstruction from Videos in the Wild via Self-supervised Scene Decomposition](Vid2Avatar.md)
> [Vid2Avatar: 3D Avatar Reconstruction from Videos in the Wild via Self-supervised Scene Decomposition (moygcc.github.io)](https://moygcc.github.io/vid2avatar/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230921171140.png)

### Im4D

[Im4D (zju3dv.github.io)](https://zju3dv.github.io/im4d/)
Im4D: High-Fidelity and Real-Time Novel View Synthesis for Dynamic Scenes

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231013171508.png)

### HumanRF

> [HumanRF: High-Fidelity Neural Radiance Fields for Humans in Motion (synthesiaresearch.github.io)](https://synthesiaresearch.github.io/humanrf/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231001165622.png)

### Neural Body

> [Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans (zju3dv.github.io)](https://zju3dv.github.io/neuralbody/)

首先在SMPL6890个顶点上定义一组潜在代码，然后
使用[Total Capture: A 3D Deformation Model for Tracking Faces, Hands, and Bodies (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4498402014756757505&noteId=2065156297063368192)
从多视图图片中获取SMPL参数$S_{t}$

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231001170255.png)

### InstantNVR

[Learning Neural Volumetric Representations of Dynamic Humans in Minutes (zju3dv.github.io)](https://zju3dv.github.io/instant_nvr/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231001172828.png)

### 4K4D

[4K4D (zju3dv.github.io)](https://zju3dv.github.io/4k4d/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023154027.png)

### D3GA

[D3GA - Drivable 3D Gaussian Avatars - Wojciech Zielonka](https://zielon.github.io/d3ga/)

多视图视频作为输入 + 3DGS + 笼形变形

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231117103528.png)

## Human-Object Interactions

### Instant-NVR

[Instant-NVR: Instant Neural Volumetric Rendering for Human-object Interactions from Monocular RGBD Stream](https://nowheretrix.github.io/Instant-NVR/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008115305.png)

### NeuralDome

[NeuralDome (juzezhang.github.io)](https://juzezhang.github.io/NeuralDome/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008120011.png)


# PIFu Occupancy Field

> [PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization](PIFu.md)
> [PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization (shunsukesaito.github.io)](https://shunsukesaito.github.io/PIFu/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230928170950.png)

## PIFuHD

> [PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization](PIFuHD.md)
> [PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization (shunsukesaito.github.io)](https://shunsukesaito.github.io/PIFuHD/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230928175323.png)

## PIFu for the Real World

[X-zhangyang/SelfPIFu--PIFu-for-the-Real-World: Dressed Human Reconstrcution from Single-view Real World Image (github.com)](https://github.com/X-zhangyang/SelfPIFu--PIFu-for-the-Real-World)
[PIFu for the Real World: A Self-supervised Framework to Reconstruct Dressed Human from Single-view Images (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4660017586591776769&noteId=1996688855483354880)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008194000.png)

## DIFu

[DIFu: Depth-Guided Implicit Function for Clothed Human Reconstruction (eadcat.github.io)](https://eadcat.github.io/DIFu/)
[DIFu: Depth-Guided Implicit Function for Clothed Human Reconstruction (thecvf.com)](https://openaccess.thecvf.com/content/CVPR2023/papers/Song_DIFu_Depth-Guided_Implicit_Function_for_Clothed_Human_Reconstruction_CVPR_2023_paper.pdf)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008114221.png)

## SeSDF

[SeSDF: Self-evolved Signed Distance Field for Implicit 3D Clothed Human Reconstruction (yukangcao.github.io)](https://yukangcao.github.io/SeSDF/)
[SeSDF: Self-evolved Signed Distance Field for Implicit 3D Clothed Human Reconstruction (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4740902287992438785&noteId=1996730143273232896)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008113348.png)

## UNIF

[UNIF: United Neural Implicit Functions for Clothed Human Reconstruction and Animation | Shenhan Qian](https://shenhanqian.github.io/unif)
[UNIF: United Neural Implicit Functions for Clothed Human Reconstruction and Animation (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4648065386802069505&noteId=1996740483288731392)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008163003.png)

## Structured 3D Features

Reconstructing **Relightable** and **Animatable** Avatars
[Enric Corona](https://enriccorona.github.io/s3f/)
[Structured 3D Features for Reconstructing Relightable and Animatable Avatars (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4700589883291336705&noteId=1996756493166029056)

X,3d fea,2d fea --> transformer --> sdf, albedo
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008174219.png)

## GTA

[Global-correlated 3D-decoupling Transformer for Clothed Avatar Reconstruction (river-zhang.github.io)](https://river-zhang.github.io/GTA-projectpage/)
[Global-correlated 3D-decoupling Transformer for Clothed Avatar Reconstruction (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4804636732393783297&noteId=2021327250504312576)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231016094412.png)

## Get3DHuman

[Get3DHuman: Lifting StyleGAN-Human into a 3D Generative Model using Pixel-aligned Reconstruction Priors. (x-zhangyang.github.io)](https://x-zhangyang.github.io/2023_Get3DHuman/)

GAN + PIFus
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023160121.png)

## DRIFu

[kuangzijian/drifu-for-animals: meta-learning based pifu model for animals (github.com)](https://github.com/kuangzijian/drifu-for-animals)

鸟类PIFu
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231123151648.png)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231123151707.png)

### SIFU

[SIFU Project Page (river-zhang.github.io)](https://river-zhang.github.io/SIFU-projectpage/)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231218204004.png)

# Depth&Normal Estimation

![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230930182135.png)

## ICON

> [ICON: Implicit Clothed humans Obtained from Normals](ICON.md)
> [ICON (mpg.de)](https://icon.is.tue.mpg.de/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230930162915.png)

## ECON

> [ECON: Explicit Clothed humans Obtained from Normals](ECON.md)
> [ECON: Explicit Clothed humans Optimized via Normal integration (xiuyuliang.cn)](https://xiuyuliang.cn/econ/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230930173026.png)

## 2K2K

DepthEstimation

> [2K2K：High-fidelity 3D Human Digitization from Single 2K Resolution Images](2K2K.md)
> [High-fidelity 3D Human Digitization from Single 2K Resolution Images Project Page (sanghunhan92.github.io)](https://sanghunhan92.github.io/conference/2K2K/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230921160120.png)

## MVSNet

DepthEstimation

> [MVSNet: Depth Inference for Unstructured Multi-view Stereo](MVSNet.md)
> [YoYo000/MVSNet: MVSNet (ECCV2018) & R-MVSNet (CVPR2019) (github.com)](https://github.com/YoYo000/MVSNet)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231002110228.png)

## GC-MVSNet

多尺度+多视图几何一致性
[GC-MVSNet: Multi-View, Multi-Scale, Geometrically-Consistent Multi-View Stereo (arxiv.org)](https://arxiv.org/abs/2310.19583)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231031172920.png)

## MonoDiffusion

[MonoDiffusion: Self-Supervised Monocular Depth Estimation Using Diffusion Model](https://arxiv.org/abs/2311.07198)

用 Diffusion Model 进行深度估计(自动驾驶)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231116153515.png)

## NDDepth

[NDDepth: Normal-Distance Assisted Monocular Depth Estimation and Completion](https://arxiv.org/abs/2311.07166)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231116153659.png)

## OccNeRF

[LinShan-Bin/OccNeRF: Code of "OccNeRF: Self-Supervised Multi-Camera Occupancy Prediction with Neural Radiance Fields". (github.com)](https://github.com/LinShan-Bin/OccNeRF)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231218200234.png)


# Other

## Texture

### Paint3D

[OpenTexture/Paint3D: Paint3D: Paint Anything 3D with Lighting-Less Texture Diffusion Models, a no lighting baked texture generative model (github.com)](https://github.com/OpenTexture/Paint3D)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231223172838.png)


## Explicit Template Decomposition

### TeCH

[TeCH: Text-guided Reconstruction of Lifelike Clothed Humans (huangyangyi.github.io)](https://huangyangyi.github.io/TeCH/)

DMTet 表示：consists of an explicit body shape grid and an implicit distance field
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231102112309.png)

### CloSET

[CloSET CVPR 2023 (liuyebin.com)](https://www.liuyebin.com/closet/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008110803.png)

### Chupa

[Chupa (snuvclab.github.io)](https://snuvclab.github.io/chupa/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008164813.png)

##  Human Face

## GPAvatar

[xg-chu/GPAvatar: [ICLR 2024] Generalizable and Precise Head Avatar from Image(s) (github.com)](https://github.com/xg-chu/GPAvatar)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240122172842.png)


### HeadRecon

[[2312.08863] HeadRecon: High-Fidelity 3D Head Reconstruction from Monocular Video (arxiv.org)](https://arxiv.org/abs/2312.08863)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231218201501.png)

### GaussianHead

[[2312.01632] GaussianHead: Impressive 3D Gaussian-based Head Avatars with Dynamic Hybrid Neural Field (arxiv.org)](https://arxiv.org/abs/2312.01632)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231205153845.png)

### GaussianAvatars

[GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians | Shenhan Qian](https://shenhanqian.github.io/gaussian-avatars)

![method.jpg|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/picturesmethod.jpg)

### TRAvatar

[Towards Practical Capture of High-Fidelity Relightable Avatars (travatar-paper.github.io)](https://travatar-paper.github.io/)

动态人脸
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231121121931.png)


### FLARE

[FLARE (mpg.de)](https://flare.is.tue.mpg.de/)

FLARE: Fast Learning of Animatable and Relightable Mesh Avatars

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231114093649.png)

### HRN

> [A Hierarchical Representation Network for Accurate and Detailed Face Reconstruction from In-The-Wild Images](HRN.md)
> [HRN (younglbw.github.io)](https://younglbw.github.io/HRN-homepage/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230921173632.png)

### 单目 3D 人脸重建

[A Perceptual Shape Loss for Monocular 3D Face Reconstruction](https://arxiv.org/abs/2310.19580)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231031181210.png)

### BakedAvatar

[BakedAvatar: Baking Neural Fields for Real-Time Head Avatar Synthesis (arxiv.org)](https://arxiv.org/pdf/2311.05521.pdf)

头部实时新视图生成
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231110155612.png)

### Video

- **3D-Aware Talking-Head Video Motion Transfer** <https://arxiv.org/abs/2311.02549>
- [Portrait4D: Learning One-Shot 4D Head Avatar Synthesis using Synthetic Data (yudeng.github.io)](https://yudeng.github.io/Portrait4D/)
- [DiffusionAvatars: Deferred Diffusion for High-fidelity 3D Head Avatars (tobias-kirschstein.github.io)](https://tobias-kirschstein.github.io/diffusion-avatars/)
- [CosAvatar (ustc3dv.github.io)](https://ustc3dv.github.io/CosAvatar/)

## Segmented Instance/Object

### Registered and Segmented Deformable Object Reconstruction from a Single View Point Cloud

[Registered and Segmented Deformable Object Reconstruction from a Single View Point Cloud](https://arxiv.org/abs/2311.07357)

配准 + 分割物体重建
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231116153203.png)

### 3DFusion, A real-time 3D object reconstruction pipeline based on streamed instance segmented data

[3DFusion, A real-time 3D object reconstruction pipeline based on streamed instance segmented data](https://arxiv.org/abs/2311.06659)

![image.png|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231116153353.png)

## Human Body Shape Completion

[Human Body Shape Completion With Implicit Shape and Flow Learning (thecvf.com)](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Human_Body_Shape_Completion_With_Implicit_Shape_and_Flow_Learning_CVPR_2023_paper.pdf)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008160354.png)

## Incomplete Image

Complete 3D Human Reconstruction from a Single Incomplete Image

[Complete 3D Human Reconstruction from a Single Incomplete Image (junyingw.github.io)](https://junyingw.github.io/paper/3d_inpainting/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008114841.png)


## New NetWork FeatER

[FeatER: An Efficient Network for Human Reconstruction via Feature Map-Based TransformER (zczcwh.github.io)](https://zczcwh.github.io/feater_page/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008160659.png)

## HF-Avatar

[hzhao1997/HF-Avatar (github.com)](https://github.com/hzhao1997/HF-Avatar?tab=readme-ov-file)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231017182026.png)

## 多模态数字人生成(数字人视频)

[An Implementation of Multimodal Fusion System for Intelligent Digital Human Generation](https://arxiv.org/pdf/2310.20251.pdf)

输入：文本、音频、图片
输出：自定义人物视频(图片/+修改/+风格化)+音频(文本合成+音频音色参考)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231101153147.png)

### IPVNet

[robotic-vision-lab/Implicit-Point-Voxel-Features-Network: Implicit deep neural network for 3D surface reconstruction. (github.com)](https://github.com/robotic-vision-lab/Implicit-Point-Voxel-Features-Network)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231108222654.png)

