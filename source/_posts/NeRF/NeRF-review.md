---
title: NeRF目前进展
date: 2023-06-25T13:01:00.000Z
tags:
  - NeRF
  - Review
categories: NeRF
date updated: 2023-07-29 13:56
---

NeRF相关的论文

<!-- more -->

# NeRF

ECCV 2020 Oral - Best Paper Honorable Mention

| Year |                                              Title&Project Page                                             | Brief Description | Conf/Jour |
| ---- | :---------------------------------------------------------------------------------------------------------: | :---------------: | :-------: |
| 2020 | [NeRF:Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf) |        初始文        |    ECCV   |

# Dalao

| PhD.School  |                                                     Homepage                                                    |                  Paper                 |
| ----------- | :-------------------------------------------------------------------------------------------------------------: | :------------------------------------: |
| HKU         |                 [刘缘Yuan Liu - Homepage (liuyuan-pal.github.io)](https://liuyuan-pal.github.io/)                 |                Neus,NeRO               |
| CUHK        |                       [胡文博HU, Wenbo's Homepage (wbhu.github.io)](https://wbhu.github.io/)                       |                Tri-MipRF               |
| HKU         | [Peng Wang (王鹏) (notion.site)](https://quartz-khaan-c6f.notion.site/Peng-Wang-0ab0a2521ecf40f5836581770c14219c) |                  Neus                  |
| UC Berkeley |                                      [Jon Barron](https://jonbarron.info/)                                      | Mip-NeRF,Mip-NeRF360,Zip-NeRF,Ref-NeRF |
| UC Berkeley |                                 [Matthew Tancik](https://www.matthewtancik.com/)                                | NerfStudio,NerfAcc,Plenoxels,Mip-NeRF, |
| UC Berkeley |                           [Ben Mildenhall (bmild.github.io)](https://bmild.github.io/)                          |                  NeRF                  |

## Surface Reconstruction

| Year |                                                            Title&Project Page                                                           |                          Brief Description                         |                               Conf/Jour                               |
| ---- | :-------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------: | :-------------------------------------------------------------------: |
| 2021 | [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction](https://lingjie0206.github.io/papers/NeuS/) |                           Neus: SDF表面重建方法                          |                                NeurIPS                                |
| 2022 |                 [Human Performance Modeling and Rendering via Neural Animated Mesh](https://zhaofuq.github.io/NeuralAM/)                |                    NSR: Neus_TSDF + NGP，但是依赖mask                   |                             SIGGRAPH Asia                             |
| 2023 |                                  [bennyguo/instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl)                                  |                       Neus+NeRF+Nerfacc+tcnn                       |                                  None                                 |
| 2023 |             [Neuralangelo: High-Fidelity Neural Surface Reconstruction](https://research.nvidia.com/labs/dir/neuralangelo/)             |                        NGP_but数值梯度+Neus_SDF                        | IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**) |
| 2023 |                                        [PermutoSDF](https://radualexandru.github.io/permuto_sdf/)                                       | NGP_butPermutohedral lattice + Neus_SDF，曲率损失和颜色MLP正则解决镜面+无纹理区域，更光滑 |                      IEEE/CVF Conference on CVPR                      |
| 2023 |                                            [NeuDA](https://3d-front-future.github.io/neuda/)                                            |                NGP_butDeformable Anchors+HPE + Neus                |                                  CVPR                                 |
| 2023 |     [NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images](https://liuyuan-pal.github.io/NeRO/)    |      Neus_SDF 新的光表示方法可以重建准确的几何和BRDF，但是细节处由于太光滑而忽略，反射颜色也依赖准确的法线     |                           SIGGRAPH (ACM TOG)                          |

## Shadow&Highlight

| Year | Title&Project Page                                                                                                               | Brief Description                                         |      Conf/Jour     |
| ---- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | :----------------: |
| 2023 | [Relighting Neural Radiance Fields with Shadow and Highlight Hints](https://nrhints.github.io/)                                  | 数据集使用相机位姿和灯源位姿                                            |      SIGGRAPH      |
| 2023 | [NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images](https://liuyuan-pal.github.io/NeRO/) | Neus_SDF 新的光表示方法可以重建准确的几何和BRDF，但是细节处由于太光滑而忽略，反射颜色也依赖准确的法线 | SIGGRAPH (ACM TOG) |

## Speed

### Encoding

| Year |                                                                  Title&Project Page                                                                 |                          Brief Description                         |                               Conf/Jour                               |
| ---- | :-------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------: | :-------------------------------------------------------------------: |
| 2022 |                   [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://nvlabs.github.io/instant-ngp/)                  |                              多分辨率哈希编码                              |                ACM Transactions on Graphics (SIGGRAPH)                |
| 2023 |                                        [bennyguo/instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl)                                        |                       Neus+NeRF+Nerfacc+tcnn                       |                                  None                                 |
| 2023 |                   [Neuralangelo: High-Fidelity Neural Surface Reconstruction](https://research.nvidia.com/labs/dir/neuralangelo/)                   |                        NGP_but数值梯度+Neus_SDF                        | IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**) |
| 2023 |                                              [PermutoSDF](https://radualexandru.github.io/permuto_sdf/)                                             | NGP_butPermutohedral lattice + Neus_SDF，曲率损失和颜色MLP正则解决镜面+无纹理区域，更光滑 |                      IEEE/CVF Conference on CVPR                      |
| 2023 |                                                  [NeuDA](https://3d-front-future.github.io/neuda/)                                                  |                NGP_butDeformable Anchors+HPE + Neus                |                                  CVPR                                 |
| 2023 |                                             [Zip-NeRF (jonbarron.info)](https://jonbarron.info/zipnerf/)                                            |        NGP + Mip-NeRF360                                                            |                                  ICCV                                 |
| 2023 | [Tri-MipRF: Tri-Mip Representation for Efficient Anti-Aliasing Neural Radiance Fields (wbhu.github.io)](https://wbhu.github.io/projects/Tri-MipRF/) |     Tri-MipRF encoding(TensoRF + NGP)+ Cone Sampling(Mip-NeRF)     |                                  ICCV                                 |

## Sampling

加速训练和渲染过程、提高新视图质量

| Year | Title&Project Page                                                          | Brief Description    | Conf/Jour |
| ---- | --------------------------------------------------------------------------- | -------------------- | :-------: |
| 2023 | [NerfAcc Documentation — nerfacc 0.5.3](https://www.nerfacc.com/en/latest/) | 一种新的采样方法可以加速NeRF     |   arXiv   |
| 2021 | [mip-NeRF ](https://jonbarron.info/mipnerf/)                                | 截头圆锥体采样方法+IPEncoding |    ICCV   |

## Large Scale Scene

| Year | Title&Project Page                                                                                            | Brief Description                      | Conf/Jour |
| ---- | ------------------------------------------------------------------------------------------------------------- | -------------------------------------- | :-------: |
| 2020 | [nerfplusplus: improves over nerf in 360 capture of unbounded scenes](https://github.com/Kai-46/nerfplusplus) | 将背景的采样点表示为四维向量，与前景分别使用不同的MLP进行训练       |   arXiv   |
| 2022 | [mip-NeRF 360](https://jonbarron.info/mipnerf360/)                                                            | 将单位球外的背景参数化，小型提议网络进行精采样，正则化dist消除小区域云雾 |    CVPR   |

## Sparse images/Generalization

| Year | Title&Project Page                                                                                                    | Brief Description    |     Conf/Jour    |
| ---- | --------------------------------------------------------------------------------------------------------------------- | -------------------- | :--------------: |
| 2022 | [SparseNeuS: Fast Generalizable Neural Surface Reconstruction from Sparse Views](https://www.xxlong.site/SparseNeuS/) | 稀疏视图重建               |       ECCV       |
| 2023 | [SparseNeRF](https://sparsenerf.github.io/)                                                                           | 利用来自现实世界不准确观测的深度先验知识 | Technical Report |
| 2021 | [pixelNeRF: Neural Radiance Fields from One or Few Images (alexyu.net)](https://alexyu.net/pixelnerf/)                |                      |       CVPR       |

## PointClouds

| Year | Title&Project Page                                                                                                                                    | Brief Description |                Conf/Jour               |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | :------------------------------------: |
| 2023 | [Globally Consistent Normal Orientation for Point Clouds by Regularizing the Winding-Number Field](https://xrvitd.github.io/Projects/GCNO/index.html) | 使得稀疏点云和薄壁点云法向量一致  | ACM Transactions on Graphics(SIGGRAPH) |
| 2023 | [Neural Kernel Surface Reconstruction](https://research.nvidia.com/labs/toronto-ai/NKSR/)                                                             | 从点云中进行表面重建        |           CVPR 2023 Highlight          |

{% note info %}
Neus的法向量通过sdf的梯度来求得，这篇Globally Consistent Normal法向量通过Winding-Number Field来规则化
{% endnote %}


## Framework

| Year | Title&Project Page                                         | Brief Description |   Conf/Jour  |
| ---- | ---------------------------------------------------------- | ----------------- | :----------: |
| 2023 | [nerfstudio](https://docs.nerf.studio/en/latest/)          | 集成现有的NeRF方法       | ACM SIGGRAPH |
| 2022 | [SDFStudio](https://autonomousvision.github.io/sdfstudio/) | 集成基于SDF的NeRF方法    |     None     |

## 有趣的应用

| Year |                            Title&Project Page                            | Brief Description | Conf/Jour |
| ---- | :----------------------------------------------------------------------: | :---------------: | --------- |
| 2023 | [Seeing the World through Your Eyes](https://world-from-eyes.github.io/) |    从人眼的倒影中重建物体    | None      |

