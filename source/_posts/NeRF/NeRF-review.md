---
title: NeRF目前进展
date: 2023-06-25T13:01:00.000Z
tags:
  - NeRF review
categories: NeRF
date updated: 2023-07-18 15:50
---

追踪一些最新的NeRF研究

<!-- more -->

NeRF++ 前背景分离
Mip-NeRF锥形光线

# NeRF

ECCV 2020 Oral - Best Paper Honorable Mention

| Year |                                              Title&Project Page                                             | Brief Description | Conf/Jour |
| ---- | :---------------------------------------------------------------------------------------------------------: | :---------------: | :-------: |
| 2020 | [NeRF:Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf) |        初始文        |    ECCV   |

## Surface Reconstruction

| Year |                                                            Title&Project Page                                                           |                          Brief Description                         |                               Conf/Jour                               |
| ---- | :-------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------: | :-------------------------------------------------------------------: |
| 2021 | [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction](https://lingjie0206.github.io/papers/NeuS/) |                           Neus: SDF表面重建方法                           |                                NeurIPS                                |
| 2022 |                 [Human Performance Modeling and Rendering via Neural Animated Mesh](https://zhaofuq.github.io/NeuralAM/)                |                   NSR: Neus_TSDF + NGP，但是依赖mask                  |                             SIGGRAPH Asia                             |
| 2023 |                                  [bennyguo/instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl)                                  |                       Neus+NeRF+Nerfacc+tcnn                       |                                  None                                 |
| 2023 |             [Neuralangelo: High-Fidelity Neural Surface Reconstruction](https://research.nvidia.com/labs/dir/neuralangelo/)             |                        NGP_but数值梯度+Neus_SDF                        | IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**) |
| 2023 |                                        [PermutoSDF](https://radualexandru.github.io/permuto_sdf/)                                       | NGP_butPermutohedral lattice + Neus_SDF，曲率损失和颜色MLP正则解决镜面+无纹理区域，更光滑 |                      IEEE/CVF Conference on CVPR                      |
| 2023 |                                            [NeuDA](https://3d-front-future.github.io/neuda/)                                            |                NGP_butDeformable Anchors+HPE + Neus                |                                  CVPR                                 |

## Speed

| Year |                                                Title&Project Page                                                | Brief Description |                Conf/Jour                |
| ---- | :--------------------------------------------------------------------------------------------------------------: | :---------------: | :-------------------------------------: |
| 2022 | [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://nvlabs.github.io/instant-ngp/) |      多分辨率哈希编码     | ACM Transactions on Graphics (SIGGRAPH) |

## Sampling

| Year | Title&Project Page                                                          | Brief Description | Conf/Jour |
| ---- | --------------------------------------------------------------------------- | ----------------- | :-------: |
| 2023 | [NerfAcc Documentation — nerfacc 0.5.3](https://www.nerfacc.com/en/latest/) | 一种新的采样方法可以加速NeRF  |   arXiv   |

## Sparse images/Generalization

| Year | Title&Project Page                                                                                                    | Brief Description    |     Conf/Jour    |
| ---- | --------------------------------------------------------------------------------------------------------------------- | -------------------- | :--------------: |
| 2022 | [SparseNeuS: Fast Generalizable Neural Surface Reconstruction from Sparse Views](https://www.xxlong.site/SparseNeuS/) | 稀疏视图重建               |       ECCV       |
| 2023 | [SparseNeRF](https://sparsenerf.github.io/)                                                                           | 利用来自现实世界不准确观测的深度先验知识 | Technical Report |

## Large Scale Scene

| Year | Title&Project Page                                                                                            | Brief Description | Conf/Jour |
| ---- | ------------------------------------------------------------------------------------------------------------- | ----------------- | :-------: |
| 2020 | [nerfplusplus: improves over nerf in 360 capture of unbounded scenes](https://github.com/Kai-46/nerfplusplus) |      将背景的采样点表示为四维向量，与前景分别使用不同的MLP进行训练             |   arXiv   |

## PointClouds

| Year | Title&Project Page                                                                                                                                    | Brief Description |                Conf/Jour               |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | :------------------------------------: |
| 2023 | [Globally Consistent Normal Orientation for Point Clouds by Regularizing the Winding-Number Field](https://xrvitd.github.io/Projects/GCNO/index.html) | 使得稀疏点云和薄壁点云法向量一致  | ACM Transactions on Graphics(SIGGRAPH) |
| 2023 | [Neural Kernel Surface Reconstruction](https://research.nvidia.com/labs/toronto-ai/NKSR/)                                                             | 从点云中进行表面重建        |           CVPR 2023 Highlight          |

{% note info %}
Neus的法向量通过sdf的梯度来求得，这篇Globally Consistent Normal法向量通过Winding-Number Field来规则化
{% endnote %}

## Shadow&Highlight

| Year | Title&Project Page                                                                              | Brief Description | Conf/Jour |
| ---- | ----------------------------------------------------------------------------------------------- | ----------------- | :-------: |
| 2023 | [Relighting Neural Radiance Fields with Shadow and Highlight Hints](https://nrhints.github.io/) | 数据集使用相机位姿和灯源位姿    |  SIGGRAPH |

## Framework

| Year | Title&Project Page                                         | Brief Description |   Conf/Jour  |
| ---- | ---------------------------------------------------------- | ----------------- | :----------: |
| 2023 | [nerfstudio](https://docs.nerf.studio/en/latest/)          | 集成现有的NeRF方法       | ACM SIGGRAPH |
| 2022 | [SDFStudio](https://autonomousvision.github.io/sdfstudio/) | 集成基于SDF的NeRF方法    |     None     |

## 有趣的应用

| Year |                            Title&Project Page                            | Brief Description | Conf/Jour |
| ---- | :----------------------------------------------------------------------: | :---------------: | --------- |
| 2023 | [Seeing the World through Your Eyes](https://world-from-eyes.github.io/) |    从人眼的倒影中重建物体    | None      |
