---
title: Paper About Multi-view 3D Reconstruction
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

>[3D Representation Methods: A Survey | PDF](https://arxiv.org/pdf/2410.06475)
>[A Review on Deep Learning Approaches for 3D Data Representations in Retrieval and Classifications](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9043500)


![image.png|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241010195323.png)



# Tools

数据集处理
- [NeuS/preprocess_custom_data at main · Totoro97/NeuS](https://github.com/Totoro97/NeuS/tree/main/preprocess_custom_data)
- Neuralangelo
  - [neuralangelo/DATA_PROCESSING.md at main · NVlabs/neuralangelo](https://github.com/NVlabs/neuralangelo/blob/main/DATA_PROCESSING.md)
  - [mli0603/BlenderNeuralangelo: Blender addon to inspect and preprocess COLMAP data for Neuralangelo (CVPR 2023)](https://github.com/mli0603/BlenderNeuralangelo)


相机外参可视化
[demul/extrinsic2pyramid: Visualize Camera's Pose Using Extrinsic Parameter by Plotting Pyramid Model on 3D Space (github.com)](https://github.com/demul/extrinsic2pyramid)

相机位姿360°视频渲染:

> https://github.com/hbb1/2d-gaussian-splatting/blob/main/render.py 可以参考这里的generate_path 把训练相机丢进去就可以fit出一个360路径进行渲染


SuperSplat 是一个免费的开源工具，用于检查和编辑 3D 高斯 Splat。它基于 Web 技术构建并在浏览器中运行，因此无需下载或安装任何内容。
https://playcanvas.com/supersplat/editor
[playcanvas/supersplat: 3D Gaussian Splat Editor](https://github.com/playcanvas/supersplat)

从sdf查询中提取表面

[SarahWeiii/diso: Differentiable Iso-Surface Extraction Package (DISO)](https://github.com/SarahWeiii/diso)
对比DMTet	FlexiCubes	DiffMC	DiffDMC

# COLMAP

> [CVPR 2017 Tutorial - Large-­scale 3D Modeling from Crowdsourced Data](https://demuc.de/tutorials/cvpr2017/) 大尺度3D modeling

3D modeling pipeline：[demuc.de/tutorials/cvpr2017/introduction1.pdf](https://demuc.de/tutorials/cvpr2017/introduction1.pdf)
- Data Association，找到图片之间的相关性
  - Descriptors：好的图片特征应该有以下特性：Repeatability, Saliency, Compactness and efficiency, Locality，也存在一些挑战——需要在different scales (sizes) and different orientations 甚至是 不同光照条件下进行探测特征
    - Global image descriptor,
      - Color histograms 颜色直方图来作为描述符
      - GIST-feature, eg: Several frequency bands and orientations for each image location. Tiling of the image, for example 4x4, and at different resolutions. Color histogram
    - Local image descriptor
      - SIFT-detector(Scale and image-plane-rotation invariant feature descriptor)
      - DSP-SIFT
      - BRIEF
      - ORB: Fast Corner Detector
- 3D points & camera pose
- Dense 3D
- Model generation

[demuc.de/tutorials/cvpr2017/sparse-modeling.pdf](https://demuc.de/tutorials/cvpr2017/sparse-modeling.pdf) 增量式的全部流程解析

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241022193306.png)

几大挑战：
- Watermarks, timestamps, frames (WTFs)
- Calibration：Focal length unknown、Image distortion
- Scale ambiguity 无法得知场景具体的尺寸
  - Use GPS EXIF tags for geo-registration
  - Use semantics to infer scale 基于语义和先验来进行赋予尺寸
- Dynamic objects
- Repetitive Structure (对称结构)
- Illumination Change
- 视图选择(初始化视图、next best view)


### COLMAP

> [colmap tutorial](https://colmap.github.io/tutorial.html)
> Document：[COLMAP — COLMAP 3.11.0.dev0 documentation](https://colmap.github.io/)
> windows pre-release[Releases · colmap/colmap](https://github.com/colmap/colmap/releases)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231005200141.png)


### HLOC

>[cvg/Hierarchical-Localization: Visual localization made easy with hloc](https://github.com/cvg/Hierarchical-Localization)
>[From Coarse to Fine: Robust Hierarchical Localization at Large Scale](https://arxiv.org/pdf/1812.03506)

![hloc.png (856×529)](https://raw.githubusercontent.com/cvg/Hierarchical-Localization/master/doc/hloc.png)


## VGGSfM

[VGGSfM: Visual Geometry Grounded Deep Structure From Motion](https://vggsfm.github.io/)
We are highly inspired by [colmap](https://github.com/colmap/colmap), [pycolmap](https://github.com/colmap/pycolmap), [posediffusion](https://github.com/facebookresearch/PoseDiffusion), [cotracker](https://github.com/facebookresearch/co-tracker), and [kornia](https://github.com/kornia/kornia).

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240617145211.png)


## GLOMAP

[colmap/glomap: GLOMAP - Global Structured-from-Motion Revisited](https://github.com/colmap/glomap?tab=readme-ov-file)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240731185452.png)



## ACE Zero

[ACE Zero](https://nianticlabs.github.io/acezero/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240808142917.png)


## XFeat

快速特征提取+匹配(轻量)

[verlab/accelerated_features: Implementation of XFeat (CVPR 2024). Do you need robust and fast local feature extraction? You are in the right place!](https://github.com/verlab/accelerated_features/tree/main?tab=readme-ov-file)

![xfeat_quali.jpg (833×764)](https://raw.githubusercontent.com/verlab/accelerated_features/refs/heads/main/figs/xfeat_quali.jpg)

# NeRF (VR+Field)

```ad-note
**形式中立+定义纯粹** NeRF阵营图
![9x9.jpg|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/9x9.jpg)

如果可以rasterize a field, 绝对是划时代的成就
![ab1a4179398fefc3f116c288c96ea17.jpg|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/ab1a4179398fefc3f116c288c96ea17.jpg)
```

| Review                                                                                                                                          |     |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| [NeRF in the industrial and robotics domain](NeRF%20in%20the%20industrial%20and%20robotics%20domain.md)                                         |     |
| [Image-Based 3D Object Reconstruction](Image-Based%203D%20Object%20Reconstruction.md)                                                           |     |
| [A Review of Deep Learning-Powered Mesh Reconstruction Methods](A%20Review%20of%20Deep%20Learning-Powered%20Mesh%20Reconstruction%20Methods.md) |     |
| [A Critical Analysis of NeRF-Based 3D Reconstruction](A%20Critical%20Analysis%20of%20NeRF-Based%203D%20Reconstruction.md)                       |     |
| [2023 Conf about NeRF](2023%20Conf%20about%20NeRF.md)                                                                                           |     |

[NeRF-review](NeRF-review.md) | [NeRF-Studio (code)](NeRF-Studio.md)

## Baseline

[NerfBaselines](https://jkulhanek.com/nerfbaselines/)

## Important Papers

| Year             | Note                                                                                                                                              | Overview                                                                                                                            | Description                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| 2020             | [DVR](DVR.md)                                                                                                                                     | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231208100556.png)                                 |                                              |
| 2020             | [IDR](IDR.md)                                                                                                                                     | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230906183157.png)                                |                                              |
| 2021             | [VolSDF](VolSDF.md)                                                                                                                               | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231201160653.png)                                 |                                              |
| 2021             | [UNISURF](UNISURF.md)                                                                                                                             | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230806143334.png)                                |                                              |
| 2021             | [NeuS](NeuS.md)                                                                                                                                   | ![Pasted image 20230531185214.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/Pasted%20image%2020230531185214.png) |                                              |
| 2022             | [HF-NeuS](HF-NeuS.md)                                                                                                                             |                                                                                                                                     |                                              |
| 2022             | [NeuralWarp](Multi-view/Implicit%20Function/NeRF-based/NeuralWarp.md)                                                                             | ![teaser.jpg (1304×708)](https://imagine.enpc.fr/~darmonf/NeuralWarp/teaser.jpg)                                                    | a direct photo-consistency term              |
| 2022             | [Geo-Neus](Geo-Neus.md)                                                                                                                           | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230911165550.png)                                | 验证了颜色渲染偏差，基于SFM稀疏点云的几何监督, 基于光度一致性监督进行几何一致性监督 |
| 2022             | [Instant-NSR](Instant-NSR.md)                                                                                                                     | ![pipeline](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/pipeline.jpg)                                      |                                              |
| 2022             | [Neus-Instant-nsr-pl](Neus-Instant-nsr-pl.md)                                                                                                     | No Paper!!! Just Code                                                                                                               |                                              |
| 2023             | [RayDF](RayDF.md)                                                                                                                                 | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231113155552.png)                                 |                                              |
| 2023             | [NISR](NISR.md)                                                                                                                                   | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230925133140.png)                                |                                              |
| 2023             | [NeUDF](NeUDF.md)                                                                                                                                 | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230825151238.png)                                |                                              |
| 2023             | [LoD-NeuS](LoD-NeuS.md)                                                                                                                           | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240104132053.png)                                 |                                              |
| 2023             | [D-NeuS](D-NeuS.md)                                                                                                                               | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230927202731.png)                                |                                              |
| 2023             | [Color-NeuS](Color-NeuS.md)                                                                                                                       | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230901123131.png)                                |                                              |
| 2023             | [BakedSDF](BakedSDF.md)                                                                                                                           | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230913154905.png)                                |                                              |
| 2023             | [Neuralangelo](Neuralangelo.md)                                                                                                                   | ![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230716140552.png)                               |                                              |
| 2023             | [NeuS2](NeuS2.md)                                                                                                                                 | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230826151714.png)                                |                                              |
| 2023             | [PermutoSDF](PermutoSDF.md)                                                                                                                       | ![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230716172220.png)                               |                                              |
| [2023](NeuDA.md) | [NeuDA](https://3d-front-future.github.io/neuda/)                                                                                                 | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240929220847.png)                        |                                              |
| 2023             | [Ref-NeuS](Ref-NeuS.md)                                                                                                                           |                                                                                                                                     |                                              |
| 2023             | [ShadowNeuS](ShadowNeuS.md)                                                                                                                       |                                                                                                                                     |                                              |
| 2023             | [NoPose-NeuS](NoPose-NeuS.md)                                                                                                                     |                                                                                                                                     |                                              |
| 2022             | [MonoSDF](MonoSDF.md)                                                                                                                             |                                                                                                                                     |                                              |
| 2022             | [RegNeRF](RegNeRF.md)                                                                                                                             |                                                                                                                                     |                                              |
|                  |                                                                                                                                                   |                                                                                                                                     |                                              |
| 2023             | [ashawkey/nerf2mesh: [ICCV2023] Delicate Textured Mesh Recovery from NeRF via Adaptive Surface Refinement](https://github.com/ashawkey/nerf2mesh) | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240717101942.png)                        |                                              |
|                  |                                                                                                                                                   |                                                                                                                                     |                                              |
| 2024             | [PSDF: Prior-Driven Neural Implicit Surface Learning for Multi-view ReconstructionPDF](https://arxiv.org/pdf/2401.12751)                          | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240823133432.png)                        | 法向量先验+多视图一致，使用深度图引导采样                        |
|                  |                                                                                                                                                   |                                                                                                                                     |                                              |


## ActiveNeRF

>[ActiveNeRF: Learning Accurate 3D Geometry by Active Pattern Projection | PDF](https://arxiv.org/pdf/2408.06592)

结合了结构光的思路，将图案投影到空间的场景/物体中来提高 NeRF 的几何质量

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240929233304.png)


## VF-NeRF

>[VF-NeRF: Learning Neural Vector Fields for Indoor Scene Reconstruction | PDF](https://arxiv.org/pdf/2408.08766)

针对室内场景的纹理较弱的区域

Neural Vector Fields 表示空间中一点到最近表面点的向量
VF 由指向最近表面点的单位矢量定义。因此，它在表面处翻转方向并等于显式表面法线。除了这种翻转之外，VF 沿平面保持不变，并在表示平面时提供强大的归纳偏差。

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240929230754.png)


## $R^2$-Mesh

>[$R^2$-Mesh: Reinforcement Learning Powered Mesh Reconstruction via Geometry and Appearance Refinement | PDF](https://arxiv.org/pdf/2408.10135)

强化学习不断优化从辐射场和SDF中提取的外观和网格，通过可微的提取策略

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240929230049.png)


## MIMO-NeRF

[MIMO-NeRF](https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/mimo-nerf/)

多输入多输出的MLP(NeRF)，减少MLP评估次数，可以加速NeRF。但是每个点的颜色和体密度根据in a group所选的输入的坐标不同，而导致歧义。$\mathcal{L}_{pi xel}$不足以处理这种歧义，引入了self-supervised learning
- Group shift. 对每个点通过多个不同的group进行评估
  - ![image.png|222](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240717135401.png)

- Variation reduction. 对每个点重复多给几次
  - ![image.png|222](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240717135412.png)


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240717085053.png)


## ActiveNeuS

[Active Neural 3D Reconstruction with Colorized Surface Voxel-based View Selection](https://arxiv.org/pdf/2405.02568)

active 3D reconstruction using uncertainty-guided view selection考虑两种不确定性的情况下评估候选视图, introduce **Colorized Surface Voxel (CSV)-based view selection**, a new next-best view (NBV) selection method exploiting surface voxel-based measurement of uncertainty in scene appearance
- 相比主动选择Nest besh view，是不是image pixel of biggest error 更好一点

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

[ReTR: Modeling Rendering via Transformer for Generalizable Neural Surface Reconstruction](https://yixunliang.github.io/ReTR/)
(Generalized Rendering Function) **使用Transformer 代替渲染过程，用了深度监督**

It introduces a learnable meta-ray token and **utilizes the cross-attention mechanism to simulate the interaction of rendering process with sampled points and render the observed color.** 
- 使用Transformer 代替渲染过程，之前方法 only use the transformer to enhance feature aggregation, and the sampled point features are still decoded into colors and densities and **aggregated using traditional volume rendering, leading to unconfident surface prediction**。作者认为提渲染简化了光线传播的过程(吸收，激发，散射)，每种光子都代表一种非线性光子-粒子相互作用，入射光子的特性受其固有性质和介质特性的影响。***体渲染公式将这些复杂性(非线性)压缩在单个密度值中，导致了入射光子建模的过度简化***。***此外颜色的权重混合忽视了复杂的物理影响，过度依赖输入视图的投影颜色***。该模型需要更广泛地累积精确表面附近各点的投影颜色，从而造成 "浑浊 "的表面外观。
- depth loss的作用更大一点，预测出来的深度图更准确
- by operating **within a high-dimensional feature space rather than the color space**, ReTR mitigates sensitivity to projected colors in source views.

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


# 3DGS (Rasterize+Primitives)

[A Survey on 3D Gaussian Splatting](https://arxiv.org/abs/2401.03890)
[Recent Advances in 3D Gaussian Splatting](https://arxiv.org/abs/2403.11134)

## Idea

### Localized Points Management

[Gaussian Splatting with Localized Points Management](https://arxiv.org/pdf/2406.04251)

局部点管理。**目的是优化新视图生成的质量**，针对生成效果不好的局部部位加强管理。(Error map，cross-view region)

但是对于NeRF这种隐式重建(or新视图生成)方法来说，增强某些弱的重建部位，可能会影响其他已经重建好的部位。**能否将弱的重建部位用另一个网络进行重建**：*得先知道哪些点是需要优化的点(error map上的点)*

- 局部优化思想

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240617101305.png)


## IGS

> [Implicit Gaussian Splatting with Efficient Multi-Level Tri-Plane Representation](https://arxiv.org/pdf/2408.10041)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241011152837.png)



## DN-Splatter

> [DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing](https://maturk.github.io/dn-splatter/)

Indoor scene，融合了来自handheld devices和general-purpose networks的深度和法向量先验

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240823125943.png)


## InstantSplat(Sparse-view)

> [InstantSplat: Sparse-view SfM-free Gaussian Splatting in Seconds](https://instantsplat.github.io/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240801174738.png)

## VCR-GauS

> [VCR-GauS](https://hlinchen.github.io/projects/VCR-GauS/)

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

# EVER (VR+Primitives)

[Exact Volumetric Ellipsoid Rendering - Alexander Mai's Homepage](https://half-potato.gitlab.io/posts/ever/)

使用Volume Rendering(NeRF) 的方式来渲染primitive based representation(3DGS)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241003162855.png)



# Depth-based

## Shakes on a Plane

[Shakes on a Plane: Unsupervised Depth Estimation from Unstabilized Photography – Princeton Computing Imaging Lab](https://light.princeton.edu/publication/soap/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240808174254.png)


## Mono-ViFI

[LiuJF1226/Mono-ViFI: [ECCV 2024] Mono-ViFI: A Unified Learning Framework for Self-supervised Single- and Multi-frame Monocular Depth Estimation](https://github.com/LiuJF1226/Mono-ViFI?tab=readme-ov-file)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240805124810.png)


# Mesh-based


## PuzzleAvatar

>[PuzzleAvatar: Assembling 3D Avatars from Personal Albums](https://arxiv.org/pdf/2405.14869)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241013212059.png)



## MagicClay

[MagicClay: Sculpting Meshes With Generative Neural Fields](https://arxiv.org/pdf/2403.02460)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241009140625.png)

- [ROAR: Robust Adaptive Reconstruction of Shapes Using Planar Projections](https://arxiv.org/pdf/2307.00690) 使用该方法将SDF更新到mesh 的local topology上
- 




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



