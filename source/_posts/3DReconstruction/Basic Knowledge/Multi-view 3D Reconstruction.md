---
title: Multi-view 3D Reconstruction
date: 2023-10-23 19:54:54
tags:
  - 3DReconstruction
categories: 3DReconstruction/Basic Knowledge
top: true
---

| 3D Reconstruction | Single-view                          | Multi-view                         |
| ----------------- | ------------------------------------ | ---------------------------------- |
| 特点              | **简单但信息不足，未见区域很难重建** | **多视图信息互补但一致性很难保证** |
| 深度估计 **DE**          | 2K2K,ECON                            | MVS,MVSNet-based                   |
| 隐式函数 **IF**          | PIFu,ICON                            | NeRF-based: NeuS,DoubleField       |
| 生成模型 **GM**            | BuilDIff, SG-GAN                     | DiffuStereo                        |
| 混合方法 **HM**          | HaP                                  | DMV3D                                   |

<!-- more -->

# Background

## Introduction

三维重建任务根据输入视图的数量可以划分为单视图重建与多视图重建。单视图重建使用单个视图作为输入，天然的会缺乏物体未见部分的信息，虽然可以使用大规模的数据集来让网络学习缺失区域的形状，但想要得到高保真的重建是非常困难的。而多视图重建
此外，一般的单视图生成式方法需要 3D 模型数据集作监督，这些模型数据需要花费大量的人工成本和时间成本，一般很难得到。

Multi-view 3D Reconstruction 按照物体规模可分：
- **Indoor** Scene Reconstruction
  - [Human Body](Multi-view%20Human%20Body%20Reconstruction.md)：DiffuStereo, DoubleField
  - Object：NeuS, Neuralangelo, Adaptive Shells
- **Outdoor** Scene Reconstruction
  - Large Scale Scene：Mip-NeRF 360

