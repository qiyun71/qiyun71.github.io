---
title: Muti-view 3D Reconstruction
date: 2023-11-16 10:08:43
tags:
  - 3DReconstruction
categories: /
---

3D Reconstruction
- Single Image ==简单但信息不足==
  - 深度估计(2K2K,ECON)
  - 生成式(BuilDIff, SG-GAN): GAN 或 DIffusion Model
  - 隐式函数(PIFu,ICON)
  - 组合方法
    - HaP: 深度估计 + SMPL 估计 + 生成式 DM
- Muti-view ==多视图信息互补但一致性很难保证==
  - 传统方法(MVS,MVSNet-based)
  - 隐式函数(NeRF-based: NeuS,DoubleField)
  - 组合方法
    - DiffuStereo: NeRF + 生成式 DM
  - 多视图融合
    - 多个视图通过深度估计得到每个视图的深度图(or点云)，然后采用多视图深度图融合(or点云配准) ==需要深度估计得到的图片具有多视图一致性==
    - 

<!-- more -->
# Background

单视图三维重建由于缺乏物体的部分信息，要想在无法观测位置的进行高精度地重建，需要靠大规模的数据集来训练网络，同时由于单视图生成式方法中需要使用大量 3D 模型进行监督，想要制作出一个可以包含真实物体所有可能情况的数据集非常困难。因此采用多视图的方法来进行三维重建，并采用基于 NeRF 方法

Muti-view 按照物体规模领域：
- HumanBody/Object Reconstruction
- Indoor Scene Reconstruction
- Large Scale Scene Reconstruction

