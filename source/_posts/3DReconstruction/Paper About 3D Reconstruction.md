---
title: Paper About Reconstruction
date: 2023-09-21T16:00:14.000Z
tags:
  - SurfaceReconstruction
  - 3DReconstruction
  - NeRF
  - MVS
  - 3DGS
categories: 3DReconstruction
date updated: 2023-11-17T10:42:38.000Z
---

NeRF-based重建方法之于前作监督的重建(新视图生成)方式，如MVS需要真实的深度图作监督，之前的包括生成式的方法需要三维模型的信息(PointCloud、Voxel、Mesh)作监督，NeRF-based方法**构建了一种自监督的重建方式，从图像中重建物体只需要用图像作监督**

NeRF将三维空间中所有点，通过MLP预测出对应的密度/SDF，**是一种连续的方法**(理论上，实际上由于计算机精度还是离散的)。至少在3D上不会由于离散方法(voxel)，而出现很大的锯齿/aliasing


**NeRF-based self-supervised 3D Reconstruction**
1. image and pose(COLMAP)
2. NeRF(NeuS) or 3DGS(SuGaR)
  1. 损失函数(对比像素颜色、深度、法向量、SDF梯度累积`<Eikonal term>`[Eikonal Equation and SDF - Lin’s site](https://marlinilram.github.io/posts/2022/06/eikonal/))
3. PointCloud后处理，根据不同用途如3D打印、有限元仿真分析、游戏assets，有许多格式mesh/FEMode/AMs

<!-- more -->

# COLMAP

## VGGSfM

[VGGSfM: Visual Geometry Grounded Deep Structure From Motion](https://vggsfm.github.io/)
We are highly inspired by [colmap](https://github.com/colmap/colmap), [pycolmap](https://github.com/colmap/pycolmap), [posediffusion](https://github.com/facebookresearch/PoseDiffusion), [cotracker](https://github.com/facebookresearch/co-tracker), and [kornia](https://github.com/kornia/kornia).

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240617145211.png)


# NeRF


```ad-note
**形式中立+定义纯粹** NeRF阵营图
![9x9.jpg|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/9x9.jpg)
```

| Review                                                                                                                                          |     |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| [NeRF in the industrial and robotics domain](NeRF%20in%20the%20industrial%20and%20robotics%20domain.md)                                         |     |
| [Image-Based 3D Object Reconstruction](Image-Based%203D%20Object%20Reconstruction.md)                                                           |     |
| [A Review of Deep Learning-Powered Mesh Reconstruction Methods](A%20Review%20of%20Deep%20Learning-Powered%20Mesh%20Reconstruction%20Methods.md) |     |
| [A Critical Analysis of NeRF-Based 3D Reconstruction](A%20Critical%20Analysis%20of%20NeRF-Based%203D%20Reconstruction.md)                       |     |
| [2023 Conf about NeRF](2023%20Conf%20about%20NeRF.md)                                                                                           |     |

[NeRF-review](NeRF-review.md) | [NeRF-Studio (code)](NeRF-Studio.md)

## Important Papers

| Year | Note                                          | Overview                                                                                                                                 | Description |
| ---- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| 2020 | [IDR](IDR.md)                                 | ![image.png\|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230906183157.png)                                |             |
| 2020 | [DVR](DVR.md)                                 | ![image.png\|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231208100556.png)                                 |             |
| 2021 | [VolSDF](VolSDF.md)                           | ![image.png\|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231201160653.png)                                 |             |
| 2021 | [UNISURF](UNISURF.md)                         | ![image.png\|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230806143334.png)                                |             |
| 2021 | [NeuS](NeuS.md)                               | ![Pasted image 20230531185214.png\|555](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230531185214.png) |             |
| 2022 | [HF-NeuS](HF-NeuS.md)                         |                                                                                                                                          |             |
| 2022 | [Geo-Neus](Geo-Neus.md)                       | ![image.png\|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230911165550.png)                                |             |
| 2022 | [Instant-NSR](Instant-NSR.md)                 | ![pipeline\|555](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/pipeline.jpg)                                      |             |
| 2022 | [Neus-Instant-nsr-pl](Neus-Instant-nsr-pl.md) | Paper No!!! Just Code                                                                                                                    |             |
| 2023 | [RayDF](RayDF.md)                             | ![image.png\|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231113155552.png)                                 |             |
| 2023 | [NISR](NISR.md)                               | ![image.png\|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230925133140.png)                                |             |
| 2023 | [NeUDF](NeUDF.md)                             | ![image.png\|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230825151238.png)                                |             |
| 2023 | [LoD-NeuS](LoD-NeuS.md)                       | ![image.png\|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240104132053.png)                                 |             |
| 2023 | [D-NeuS](D-NeuS.md)                           | ![image.png\|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230927202731.png)                                |             |
| 2023 | [Color-NeuS](Color-NeuS.md)                   | ![image.png\|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230901123131.png)                                |             |
| 2023 | [BakedSDF](BakedSDF.md)                       | ![image.png\|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230913154905.png)                                |             |
| 2023 | [Neuralangelo](Neuralangelo.md)               | ![image.png\|555](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230716140552.png)                               |             |
| 2023 | [NeuS2](NeuS2.md)                             | ![image.png\|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230826151714.png)                                |             |
| 2023 | [PermutoSDF](PermutoSDF.md)                   | ![image.png\|555](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230716172220.png)                               |             |
|      |                                               |                                                                                                                                          |             |
| 2023 | [Ref-NeuS](Ref-NeuS.md)                       |                                                                                                                                          |             |
| 2023 | [ShadowNeuS](ShadowNeuS.md)                   |                                                                                                                                          |             |
| 2023 | [NoPose-NeuS](NoPose-NeuS.md)                 |                                                                                                                                          |             |
| 2022 | [MonoSDF](MonoSDF.md)                         |                                                                                                                                          |             |
| 2022 | [RegNeRF](RegNeRF.md)                         |                                                                                                                                          |             |
|      |                                               |                                                                                                                                          |             |
|      |                                               |                                                                                                                                          |             |
|      |                                               |                                                                                                                                          |             |

## ActiveNeuS

[[2405.02568] ActiveNeuS: Active 3D Reconstruction using Neural Implicit Surface Uncertainty](https://arxiv.org/abs/2405.02568)

考虑两种不确定性的情况下评估候选视图
![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240508085101.png)


## Binary Opacity Grids

Binary Opacity Grids: Capturing Fine Geometric Detail for Mesh-Based View Synthesis
https://binary-opacity-grid.github.io/

推进BakedSDF的工作


## RNb-NeuS

[bbrument/RNb-NeuS: Code release for RNb-NeuS. (github.com)](https://github.com/bbrument/RNb-NeuS)

将**反射率**和**法线贴图**无缝集成为基于神经体积渲染的 3D 重建中的输入数据
考虑高光和阴影：显著改善了高曲率或低可见度区域的详细 3D 重建
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231205152551.png)

## ReTR

[Rethinking Rendering in Generalizable Neural Surface Reconstruction: A Learning-based Solution (yixunliang.github.io)](https://yixunliang.github.io/ReTR/)
修改论文 title：ReTR: Modeling Rendering via Transformer for Generalizable Neural Surface Reconstruction

CNN + 3D Decoder + **Transformer** + NeRF 用深度图监督

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231109094904.png)



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

## PonderV2

[OpenGVLab/PonderV2: PonderV2: Pave the Way for 3D Foundation Model with A Universal Pre-training Paradigm (github.com)](https://github.com/OpenGVLab/PonderV2)

PointCloud 提取特征(点云编码器) + NeRF 渲染图片 + 图片损失优化点云编码器

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231107153921.png)

## Spiking NeRF

Spiking NeRF: Representing the Real-World Geometry by a Discontinuous Representation

MLP 是连续函数，对 NeRF 网络结构的改进来生成不连续的密度场

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231116155214.png)

## Hyb-NeRF 多分辨率混合编码

[[2311.12490] Hyb-NeRF: A Multiresolution Hybrid Encoding for Neural Radiance Fields (arxiv.org)](https://arxiv.org/abs/2311.12490)

多分辨率混合编码

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231123143210.png)

## Dynamic

### In-Hand 3D Object

[In-Hand 3D Object Reconstruction from a Monocular RGB Video](https://arxiv.org/abs/2312.16425)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240104173205.png)

### DynamicSurf

[DynamicSurf: Dynamic Neural RGB-D Surface Reconstruction with an Optimizable Feature Grid](https://arxiv.org/abs/2311.08159)

单目 RGBD 视频重建 3D

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231116154921.png)


### MorpheuS

[MorpheuS (hengyiwang.github.io)](https://hengyiwang.github.io/projects/morpheus)
MorpheuS: Neural Dynamic 360° Surface Reconstruction from **Monocular RGB-D Video**
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231204133746.png)

### NPGs

[[2312.01196] Neural Parametric Gaussians for Monocular Non-Rigid Object Reconstruction (arxiv.org)](https://arxiv.org/abs/2312.01196)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231205153011.png)


# 3DGS(PointCloud-based)

[A Survey on 3D Gaussian Splatting](https://arxiv.org/abs/2401.03890)
[Recent Advances in 3D Gaussian Splatting](https://arxiv.org/abs/2403.11134)

## Idea

### Localized Points Management

[Gaussian Splatting with Localized Points Management](https://arxiv.org/pdf/2406.04251)

局部点管理。**目的是优化新视图生成的质量**，针对生成效果不好的局部部位加强管理。(Error map，cross-view region)

但是对于NeRF这种隐式重建(or新视图生成)方法来说，增强某些弱的重建部位，可能会影响其他已经重建好的部位。**能否将弱的重建部位用另一个网络进行重建**：*得先知道哪些点是需要优化的点(error map上的点)*

- 局部优化思想

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240617101305.png)


## VCR-GauS

[VCR-GauS](https://hlinchen.github.io/projects/VCR-GauS/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240618210910.png)


## PGSR

[PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction](https://zju3dv.github.io/pgsr/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240618210549.png)



## TrimGS

[TrimGS](https://trimgs.github.io/)

- Gaussian Trimming
- Scale-driven Densification
- Normal Regularization

## SuGaR

[SuGaR (enpc.fr)](https://imagine.enpc.fr/~guedona/sugar/)

3D Gaussian Splatting 提取mesh
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231123142658.png)

## 2DGS

[hbb1/2d-gaussian-splatting: [SIGGRAPH'24] 2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://github.com/hbb1/2d-gaussian-splatting)

and unofficial implementation [TimSong412/2D-surfel-gaussian](https://github.com/TimSong412/2D-surfel-gaussian)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240430084907.png)


## GS2mesh

[Surface Reconstruction from Gaussian Splatting via Novel Stereo Views](https://gs2mesh.github.io/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240419144938.png)

# Mesh-based

## MeshAnything

[MeshAnything](https://buaacyw.github.io/mesh-anything/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240617110057.png)

与 MeshGPT 等直接生成艺术家创建的网格Artist-Created Meshes (AMs)的方法相比，我们的方法**避免了学习复杂的 3D 形状分布**。相反，**它专注于通过优化的拓扑有效地构建形状**，从而显着减轻训练负担并增强可扩展性。


## Pixel2Mesh

[Pixel2Mesh (nywang16.github.io)](https://nywang16.github.io/p2m/index.html)
[Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4518061932765929473&noteId=2082454818072408064)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240130141333.png)

我们提出了一种端端深度学习架构，可以从单色图像中生成三角形网格中的3D形状。受深度神经网络特性的限制，以前的方法通常是用体积或点云表示三维形状，将它们转换为更易于使用的网格模型是很困难的。与现有方法不同，我们的网络在基于图的卷积神经网络中表示3D网格，并通过逐步变形椭球来产生正确的几何形状，利用从输入图像中提取的感知特征。我们采用了从粗到精的策略，使整个变形过程稳定，并定义了各种网格相关的损失来捕捉不同层次的属性，以保证视觉上的吸引力和物理上的精确3D几何。大量的实验表明，我们的方法不仅可以定性地产生具有更好细节的网格模型，而且与目前的技术相比，可以实现更高的3D形状估计精度。


## Pixel2Mesh++ 

[Pixel2Mesh++ (walsvid.github.io)](https://walsvid.github.io/Pixel2MeshPlusPlus/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240204211248.png)


# Voxel-based

## Voxurf

[wutong16/Voxurf: [ ICLR 2023 Spotlight ] Pytorch implementation for "Voxurf: Voxel-based Efficient and Accurate Neural Surface Reconstruction" (github.Com)](https://github.com/wutong16/Voxurf)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023163503.png)

