---
title: NeRF目前进展
date: 2023-06-25T13:01:00.000Z
tags:
  - NeRF review
categories: NeRF
date updated: 2023-06-25 13:09
---

追踪一些最新的NeRF研究

<!-- more -->
# 最新进展

## 应用方面

| Year |                                          Title&Project Page                                          |   Brief Description    |
| ---- |:----------------------------------------------------------------------------------------------------:|:----------------------:|
| 2023 | [Seeing the World through Your Eyes (world-from-eyes.github.io)](https://world-from-eyes.github.io/) | 从人眼的倒影中重建物体 |                                                                                                      |                        |

# 之前研究

| Year |                                                                       Title&Project Page                                                                        | Brief Description |
| ---- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------:|
| 2020 |                                     [NeRF: Neural Radiance Fields (matthewtancik.com)](https://www.matthewtancik.com/nerf)                                      |      初始文       |
| 2021 | [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction (lingjie0206.github.io)](https://lingjie0206.github.io/papers/NeuS/) |  SDF表面重建方法  |
| 2022 |               [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding (nvlabs.github.io)](https://nvlabs.github.io/instant-ngp/)               | 多分辨率哈希编码  |
| 2022 |   [Human Performance Modeling and Rendering via Neural Animated Mesh (zhaofuq.github.io)](https://zhaofuq.github.io/NeuralAM/)                                                                                                                                                              |   快速表面重建             |


# 个人理解
## 20230705-NeRF_Neus_InstantNGP

基于NeRF的方法主要包括以下部分：
- 神经网络结构-->训练出来模型
- 位置编码方式
- 体渲染函数
    - 不透明度，累计透光率，权重，颜色
- 采样点的采样方式(精采样)
- 光线的生成方式，near和far的计算方式

在NeRF的基础上生成mesh模型：需要确定物体的表面，用不同的方法可以生成不同的隐式模型，如NeRF为位置转密度颜色，Neus为位置转SDF。以空间原点为中心，根据bound_min和bound_max生成一个resolution x resolution x resolution立方点云模型，根据隐式模型，生成其中每个点的密度颜色或者sdf值，然后选择零水平集为物体的表面，根据物体表面上的点生成三角形网格，并得到mesh模型。


