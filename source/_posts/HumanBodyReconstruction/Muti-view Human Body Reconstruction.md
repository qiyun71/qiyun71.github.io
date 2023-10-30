---
title: Muti-view Human Body Reconstruction
date: 2023-10-09 16:33:31
tags:
  - ClothedHumans
  - 3DReconstruction
  - PointCloud
categories: HumanBodyReconstruction
---

Terminology

- Human Radiance Fields
- 3D **Clothed Human** Reconstruction | **Digitization**

Application：
- 模型获取方式：手持扫描仪、360 度相机矩阵（成本高）
- [复刻一个迷你版的自己！3d打印真人手办“火起来”，背后的故事也很动人 (yangtse.com)](https://www.yangtse.com/content/1604507html)

<!-- more -->

From [Other Paper Reconstruction](/HumanBodyReconstruction/Other%20Paper%20Reconstruction) (Depth&Normal Estimation | Implicit Function) and [Generative approach](/HumanBodyReconstruction/Generative%20approach/Generative%20approach%20Review)

Muti-view input **Idea**

1. **Depth&Normal Estimation**(2K2K)
2. ~~**Implicit Function**(PIFu or NeRF)~~
3. **Generative approach** : 2D Image --> （Encoder --> Decoder） --> 3D Mesh

**方法 1**：多视图深度图融合、多视图点云配准
(2K2K-based i.e.Depth&Normal Estimation)

- 多视图深度图融合：[DepthFusion: Fuse multiple depth frames into a point cloud](https://github.com/touristCheng/DepthFusion)
  - 需要相机位姿，位姿估计有误差
  - 更准确的位姿: BA(Bundle Adjusted 光束法平差，优化相机 pose 和 landmark)

- 多视图点云配准：[Point Cloud Registration](/HumanBodyReconstruction/PointCloud/PointCloud%20Review)
  - **点云配准**(Point Cloud Registration) 2K 生成的多角度点云形状不统一 

**方法 2**：生成式，从图片生成点云
Generative approach(Muti-view image、pose (keypoints)... --> PointCloud)

- 参考 [BuilDiff](https://github.com/weiyao1996/BuilDiff)，构建网络([PVCNNs](https://readpaper.com/pdf-annotate/note?pdfId=4544669809538392065&noteId=2018413897297176576) 单类训练)
  - 是否更换扩散网络 [DiT-3D](https://dit-3d.github.io/)，可以学习显式的类条件嵌入(生成多样化的点云)
  - 是否依靠 SMPL，根据 LBS(Linear Blending Skinning)将人体 mesh 变形到规范化空间
    - [Video2Avatar](https://moygcc.github.io/vid2avatar/) 将整个人体规范化后采样
    - [EVA3D](https://hongfz16.github.io/projects/EVA3D) 将 NeRF 融入 GAN 生成图片，并与真实图片一同训练判别器(人体规范化后分块 NeRF)

# 三维重建方法对比

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
## Implicit Function

![NeuS2.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231024153406.png)
预测每个采样点 sdf 和 feature 向量
$(sdf,\mathbf{feature})=f_\Theta(\mathbf{e}),\quad\mathbf{e}=(\mathbf{x},h_\Omega(\mathbf{x})).$

预测每个采样点颜色值
$\mathbf c=c_{\Upsilon}(\mathbf x,\mathbf n,\mathbf v,sdf,\mathbf{feature})$，$\mathbf n=\nabla_\mathbf x sdf.$

体渲染像素颜色
$\hat{C}=\sum_{i=1}^n T_i\alpha_i c_i$， $T_i=\prod_{j=1}^{i-1}(1-\alpha_j)$ ，$\alpha_i=\max\left(\frac{\Phi_s(f(\mathbf{p}(t_i))))-\Phi_s(f(\mathbf{p}(t_{i+1})))}{\Phi_s(f(\mathbf{p}(t_i)))},0\right)$

训练得到 MLP，根据 MarchingCube 得到点云
## Generative approach

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021114740.png)

![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231024111221.png)

