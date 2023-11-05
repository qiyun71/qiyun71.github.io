---
title: Other Paper
date: 2023-09-21 16:00:14
tags:
  - SurfaceReconstruction
  - 3DReconstruction
  - NeRF
  - MVS
  - PIFu
categories: HumanBodyReconstruction
---

Awesome Human Body Reconstruction

1. **Depth&Normal Estimation**(2K2K)
2. **Implicit Function**(PIFu or NeRF)

| Method | 泛化   | 数据集监督                              | 提取 mesh 方式                   | 获得纹理方式            |
| ------ | ------ | --------------------------------------- | ------------------------------ | ----------------------- |
| 2k2k   | 比较好 | (mesh+texture:)depth、normal、mask、rgb | 高质量深度图 --> 点云 --> mesh | 图片 rgb 贴图 |
| PIFu   | 比较好 | 点云(obj)、rgb(uv)、mask、camera        | 占用场 --> MC --> 点云,mesh    | 表面颜色场              |
| NeRF   | 差     | rgb、camera                             | 密度场 --> MC --> 点云,mesh    | 体积颜色场              |
| NeuS   | 差     | rgb、camera                             | SDF --> MC --> 点云,mesh       | 体积颜色场              |
| ICON   | 非常好 | rgb+mask、SMPL、法向量估计器 DR          | 占用场 --> MC --> 点云,mesh    | 图片 rgb 贴图             |
| ECON   | 非常好 | rgb+mask、SMPL、法向量估计器 DR          | d-BiNI + SC(shape completion)  | 图片 rgb 贴图             |

<!-- more -->

![Human.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/Human.png)

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

# NeRF Human Body

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

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231001170255.png)

### InstantNVR

[Learning Neural Volumetric Representations of Dynamic Humans in Minutes (zju3dv.github.io)](https://zju3dv.github.io/instant_nvr/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231001172828.png)

### 4K4D

[4K4D (zju3dv.github.io)](https://zju3dv.github.io/4k4d/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023154027.png)


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

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023160121.png)

# Explicit Template Decomposition

## TeCH

[TeCH: Text-guided Reconstruction of Lifelike Clothed Humans (huangyangyi.github.io)](https://huangyangyi.github.io/TeCH/)

DMTet 表示：consists of an explicit body shape grid and an implicit distance field
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231102112309.png)

## CloSET

[CloSET CVPR 2023 (liuyebin.com)](https://www.liuyebin.com/closet/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008110803.png)

## Chupa

[Chupa (snuvclab.github.io)](https://snuvclab.github.io/chupa/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008164813.png)

# Other

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

## Other paper about camera pose

- **USB-NeRF: Unrolling Shutter Bundle Adjusted Neural Radiance Fields** https://arxiv.org/abs/2310.02687
- How to turn your camera into a perfect pinhole model

# Human Face

## HRN

> [A Hierarchical Representation Network for Accurate and Detailed Face Reconstruction from In-The-Wild Images](HRN.md)
> [HRN (younglbw.github.io)](https://younglbw.github.io/HRN-homepage/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230921173632.png)

## 单目 3D 人脸重建

[A Perceptual Shape Loss for Monocular 3D Face Reconstruction](https://arxiv.org/abs/2310.19580)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231031181210.png)

# NeRF OtherObject Reconstruction

## NISR

> [Improving Neural Indoor Surface Reconstruction with Mask-Guided Adaptive Consistency Constraints](NISR.md)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230925133140.png)

## D-NeuS

> [Recovering Fine Details for Neural Implicit Surface Reconstruction](D-NeuS.md)
> [fraunhoferhhi/D-NeuS: Recovering Fine Details for Neural Implicit Surface Reconstruction (WACV2023) (github.com)](https://github.com/fraunhoferhhi/D-NeuS)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230927202731.png)

![15fcd4e5b38213b428a4fe32a140bf88_.jpg|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/15fcd4e5b38213b428a4fe32a140bf88_.jpg)

## Voxurf

[wutong16/Voxurf: [ ICLR 2023 Spotlight ] Pytorch implementation for "Voxurf: Voxel-based Efficient and Accurate Neural Surface Reconstruction" (github.Com)]( https://github.com/wutong16/Voxurf )

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023163503.png)

## AutoRecon

[AutoRecon: Automated 3D Object Discovery and Reconstruction (zju3dv.github.io)](https://zju3dv.github.io/autorecon/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023164638.png)


## G-Shell 

重建水密物体+衣服等非水密物体——通用
[G-Shell (gshell3d.github.io)](https://gshell3d.github.io/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231024094322.png)

# Human Body Shape Completion

[Human Body Shape Completion With Implicit Shape and Flow Learning (thecvf.com)](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Human_Body_Shape_Completion_With_Implicit_Shape_and_Flow_Learning_CVPR_2023_paper.pdf)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008160354.png)

# Incomplete Image

Complete 3D Human Reconstruction from a Single Incomplete Image

[Complete 3D Human Reconstruction from a Single Incomplete Image (junyingw.github.io)](https://junyingw.github.io/paper/3d_inpainting/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231008114841.png)

