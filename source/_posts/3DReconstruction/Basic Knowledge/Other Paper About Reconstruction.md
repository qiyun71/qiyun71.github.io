---
title: Other Paper About Reconstruction
date: 2023-09-21T16:00:14.000Z
tags:
  - SurfaceReconstruction
  - 3DReconstruction
  - NeRF
  - MVS
  - PIFu
categories: 3DReconstruction/Basic Knowledge
date updated: 2023-11-17T10:42:38.000Z
---

Awesome Human Body Reconstruction

1. **Depth&Normal Estimation**(2K2K)
2. **Implicit Function**(PIFu or NeRF)

| Method | 泛化  | 数据集监督                                | 提取 mesh 方式                    | 获得纹理方式    |
| ------ | --- | ------------------------------------ | ----------------------------- | --------- |
| 2k2k   | 比较好 | (mesh+texture:)depth、normal、mask、rgb | 高质量深度图 --> 点云 --> mesh        | 图片 rgb 贴图 |
| PIFu   | 比较好 | 点云(obj)、rgb(uv)、mask、camera          | 占用场 --> MC --> 点云,mesh        | 表面颜色场     |
| NeRF   | 差   | rgb、camera                           | 密度场 --> MC --> 点云,mesh        | 体积颜色场     |
| NeuS   | 差   | rgb、camera                           | SDF --> MC --> 点云,mesh        | 体积颜色场     |
| ICON   | 非常好 | rgb+mask、SMPL、法向量估计器 DR              | 占用场 --> MC --> 点云,mesh        | 图片 rgb 贴图 |
| ECON   | 非常好 | rgb+mask、SMPL、法向量估计器 DR              | d-BiNI + SC(shape completion) | 图片 rgb 贴图 |

<!-- more -->

# Gaussian Splatting Method

## SuGaR

[SuGaR (enpc.fr)](https://imagine.enpc.fr/~guedona/sugar/)

3D Gaussian Splatting 提取mesh
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231123142658.png)


# NeRF Other Object Reconstruction

## RNb-NeuS

[bbrument/RNb-NeuS: Code release for RNb-NeuS. (github.com)](https://github.com/bbrument/RNb-NeuS)

将**反射率**和**法线贴图**无缝集成为基于神经体积渲染的 3D 重建中的输入数据
考虑高光和阴影：显著改善了高曲率或低可见度区域的详细 3D 重建
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231205152551.png)

## Voxurf

[wutong16/Voxurf: [ ICLR 2023 Spotlight ] Pytorch implementation for "Voxurf: Voxel-based Efficient and Accurate Neural Surface Reconstruction" (github.Com)](https://github.com/wutong16/Voxurf)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023163503.png)

## ReTR

[Rethinking Rendering in Generalizable Neural Surface Reconstruction: A Learning-based Solution (yixunliang.github.io)](https://yixunliang.github.io/ReTR/)
修改论文 title：ReTR: Modeling Rendering via Transformer for Generalizable Neural Surface Reconstruction

CNN + 3D Decoder + Transformer + NeRF 用深度图监督

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231109094904.png)

## NISR

> [Improving Neural Indoor Surface Reconstruction with Mask-Guided Adaptive Consistency Constraints](NISR.md)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230925133140.png)

## D-NeuS

> [Recovering Fine Details for Neural Implicit Surface Reconstruction](D-NeuS.md)
> [fraunhoferhhi/D-NeuS: Recovering Fine Details for Neural Implicit Surface Reconstruction (WACV2023) (github.com)](https://github.com/fraunhoferhhi/D-NeuS)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230927202731.png)

![15fcd4e5b38213b428a4fe32a140bf88_.jpg|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/15fcd4e5b38213b428a4fe32a140bf88_.jpg)

## AutoRecon

[AutoRecon: Automated 3D Object Discovery and Reconstruction (zju3dv.github.io)](https://zju3dv.github.io/autorecon/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023164638.png)

## G-Shell

重建水密物体+衣服等非水密物体——通用
[G-Shell (gshell3d.github.io)](https://gshell3d.github.io/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231024094322.png)

## Adaptive Shells

[Adaptive Shells for Efficient Neural Radiance Field Rendering (nvidia.com)](https://research.nvidia.com/labs/toronto-ai/adaptive-shells/)

自适应使用基于体积的渲染和基于表面的渲染
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231117150239.png)

## DynamicSurf

[DynamicSurf: Dynamic Neural RGB-D Surface Reconstruction with an Optimizable Feature Grid](https://arxiv.org/abs/2311.08159)

单目 RGBD 视频重建 3D

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231116154921.png)

## RayDF

[RayDF: Neural Ray-surface Distance Fields with Multi-view Consistency (vlar-group.github.io)](https://vlar-group.github.io/RayDF.html)
[RayDF: Neural Ray-surface Distance Fields with Multi-view Consistency (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=2037229054391691776&noteId=2047746094923644416)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231113155552.png)

## PonderV2

[OpenGVLab/PonderV2: PonderV2: Pave the Way for 3D Foundation Model with A Universal Pre-training Paradigm (github.com)](https://github.com/OpenGVLab/PonderV2)

PointCloud 提取特征(点云编码器) + NeRF 渲染图片 + 图片损失优化点云编码器

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231107153921.png)

## Spiking NeRF

Spiking NeRF: Representing the Real-World Geometry by a Discontinuous Representation

MLP 是连续函数，对 NeRF 网络结构的改进来生成不连续的密度场

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231116155214.png)

## Hyb-NeRF

[[2311.12490] Hyb-NeRF: A Multiresolution Hybrid Encoding for Neural Radiance Fields (arxiv.org)](https://arxiv.org/abs/2311.12490)

多分辨率混合编码

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231123143210.png)

## Dynamic
### MorpheuS

[MorpheuS (hengyiwang.github.io)](https://hengyiwang.github.io/projects/morpheus)
MorpheuS: Neural Dynamic 360° Surface Reconstruction from **Monocular RGB-D Video**
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231204133746.png)

### NPGs

[[2312.01196] Neural Parametric Gaussians for Monocular Non-Rigid Object Reconstruction (arxiv.org)](https://arxiv.org/abs/2312.01196)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231205153011.png)

# NeRF Human Body Reconstruction

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

# Other

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

