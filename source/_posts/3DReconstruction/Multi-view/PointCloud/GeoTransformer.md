---
title: GeoTransformer
date: 2023-10-18 17:10:29
tags:
  - PointCloud
  - Registration
categories: 3DReconstruction/Multi-view/PointCloud
---

| Title     | Geometric Transformer for Fast and Robust Point Cloud Registration                                                                                                                 |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Zheng Qin1 Hao Yu2 Changjian Wang1 Yulan Guo1,3 Yuxing Peng1 Kai Xu1*                                                                                                                                                                                   |
| Conf/Jour | CVPR                                                                                                                                                                                   |
| Year      | 2022                                                                                                                                                                                   |
| Project   | [qinzheng93/GeoTransformer: [CVPR2022] Geometric Transformer for Fast and Robust Point Cloud Registration (github.com)](https://github.com/qinzheng93/GeoTransformer)              |
| Paper     | [Geometric Transformer for Fast and Robust Point Cloud Registration (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4667181912255643649&noteId=2010116153877133824) |

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231018113631.png)

pairwise registration models, only **Ubuntu**

<!-- more -->

# Abstract

研究了点云配准中精确对应点的提取问题。最近的无关键点方法绕过了在低重叠情况下难以检测可重复关键点的方法，在配准方面显示出巨大的潜力。他们在下采样的超点上寻找对应点，然后将其传播到密集点。根据相邻补丁是否重叠来匹配重叠点。这种稀疏和松散的匹配需要上下文特征捕捉点云的几何结构。我们提出几何变压器来学习几何特征，以实现鲁棒的重叠点匹配。它对成对距离和三重角度进行编码，使其在低重叠情况下具有鲁棒性，并且对刚性变换不变性。简单的设计获得了惊人的匹配精度，在估计对准变换时不需要RANSAC，从而获得了100倍的加速度。在具有挑战性的3DLoMatch基准测试中，我们的方法将初始比率提高了17 ~ 30个百分点，将注册召回率提高了7个百分点以上。