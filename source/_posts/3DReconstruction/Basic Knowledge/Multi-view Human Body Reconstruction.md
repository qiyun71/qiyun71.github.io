---
title: Multi-view Human Body Reconstruction
date: 2023-10-09T16:33:31.000Z
tags:
  - ClothedHumans
  - 3DReconstruction
  - PointCloud
categories: 3DReconstruction/Basic Knowledge
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

1. **Depth&Normal Estimation**(2K2K) [Other Paper About Reconstruction](Other%20Paper%20About%20Reconstruction.md)
2. ~~**Implicit Function**(PIFu or NeRF)~~ [Other Paper About Reconstruction](Other%20Paper%20About%20Reconstruction.md)
3. **Generative approach**  [Generative approach](Generative%20Models%20Review.md)

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

# Paper

姿势估计可能有多种解决方案，但不准确的姿势可能会导致低分辨率的几何形状
