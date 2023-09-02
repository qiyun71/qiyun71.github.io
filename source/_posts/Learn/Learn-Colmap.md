---
title: Learn-Colmap
date: 2023-08-23 17:00:00
tags:
  - Colmap
categories: Learn
---

[colmap tutorial](https://colmap.github.io/tutorial.html)

<!-- more -->

Colmap即SFM和Multi-View Stereo

- Structure-from-Motion Revisited
    - Feature detection and extraction
    - Feature matching and geometric verification
    - Structure and motion reconstruction
- Pixelwise View Selection for Unstructured Multi-View Stereo
    - get a dense point cloud

# SFM

[SFM算法原理初简介 | jiajie (gitee.io)](https://jiajiewu.gitee.io/post/tech/slam-sfm/sfm-intro/)

![image.png|500](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230717204531.png)

SFM - 稀疏重建: 
- 特征提取 , ref: [非常详细的sift算法原理解析_可时间倒数了的博客-CSDN博客](https://blog.csdn.net/u010440456/article/details/81483145)
    - eg：sift算法(Scale-invariant feature transform)是一种电脑视觉的算法用来侦测与描述影像中的局部性特征，它在空间尺度中寻找极值点，并提取出其位置、尺度、旋转不变量，此算法由 David Lowe在1999年所发表，2004年完善总结
- 特征匹配，ref: [sfm流程概述_神气爱哥的博客-CSDN博客](https://blog.csdn.net/qingcaichongchong/article/details/62424661)

