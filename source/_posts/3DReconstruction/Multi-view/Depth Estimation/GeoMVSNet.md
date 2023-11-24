---
title: GeoMVSNet
date: 2023-10-05 09:57:23
tags:
  - MVS
categories: 3DReconstruction/Multi-view/Depth Estimation
---

| Title     | GeoMVSNet: Learning Multi-View Stereo With Geometry Perception                                                                                                                                                                      |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Author    | Zhang, Zhe and Peng, Rui and Hu, Yuxi and Wang, Ronggang                                                                                                                       |
| Conf/Jour | CVPR                                                                                                                                                                           |
| Year      | 2023                                                                                                                                                                           |
| Project   | [doubleZ0108/GeoMVSNet: [CVPR 23'] GeoMVSNet: Learning Multi-View Stereo with Geometry Perception (github.com)](https://github.com/doubleZ0108/GeoMVSNet)                      |
| Paper     | [GeoMVSNet: Learning Multi-View Stereo with Geometry Perception (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=1807465782777293568&noteId=1990827633705815808) |

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231005100022.png)

<!-- more -->

# Abstract

最近的级联多视图立体 (MVS) 方法可以通过缩小假设范围来有效地估计高分辨率深度图。然而，以往的方法忽略了粗阶段嵌入的重要几何信息，导致代价匹配脆弱，重构结果次优。在本文中，我们提出了一个几何感知模型，称为GeoMVSNet，以显式地整合粗阶段隐含的几何线索进行精细深度估计。特别是，我们设计了一个双分支几何融合网络，从粗估计中提取几何先验，以增强更精细阶段的结构特征提取。此外，我们将编码有价值的深度分布属性的粗概率体积嵌入到轻量级正则化网络中，以进一步加强深度几何直觉。同时，我们应用频域滤波来减轻高频区域的负面影响，并采用课程学习策略逐步提升模型的几何集成。为了增强我们模型的全场景几何感知，我们提出了基于高斯混合模型假设的深度分布相似性损失。在DTU和Tanks和Temples (T&T)数据集上的大量实验表明，我们的GeoMVSNet实现了最先进的结果，并在T&T-Advanced集上排名第一