---
title: Geo-Neus
date: 2023-09-04 15:54:53
tags:
  - Neus
  - SurfaceReconstruction
categories: NeRF/SurfaceReconstruction
---

| Title     | Geo-Neus: Geometry-Consistent Neural Implicit Surfaces Learning for Multi-view Reconstruction                                                                                                                 |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Fu, Qiancheng and Xu, Qingshan and Ong, Yew-Soon and Tao, Wenbing                                                                                                                                             |
| Conf/Jour | NeurIPS                                                                                                                                                                                                       |
| Year      | 2022                                                                                                                                                                                                              |
| Project   | [GhiXu/Geo-Neus: Geo-Neus: Geometry-Consistent Neural Implicit Surfaces Learning for Multi-view Reconstruction (NeurIPS 2022) (github.com)](https://github.com/GhiXu/Geo-Neus)                                |
| Paper     | [Geo-Neus: Geometry-Consistent Neural Implicit Surfaces Learning for Multi-view Reconstruction (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4629958250540843009&noteId=1943200084633808128) |

<!-- more -->

# AIR

近年来，通过体绘制的神经隐式曲面学习成为多视图重建的热门方法。然而，一个关键的挑战仍然存在: **现有的方法缺乏明确的多视图几何约束，因此通常无法生成几何一致的表面重建**。为了解决这一挑战，我们提出了几何一致的神经隐式曲面学习用于多视图重建。我们从理论上分析了体绘制积分与基于点的有符号距离函数(SDF)建模之间的差距。为了弥补这一差距，我们直接定位SDF网络的零级集，并通过利用多视图立体中的结构来自运动的稀疏几何(SFM)和光度一致性显式地执行多视图几何优化。这使得我们的SDF优化无偏，并允许多视图几何约束专注于真正的表面优化。大量实验表明，我们提出的方法在复杂薄结构和大面积光滑区域都能实现高质量的表面重建，从而大大优于目前的技术水平。