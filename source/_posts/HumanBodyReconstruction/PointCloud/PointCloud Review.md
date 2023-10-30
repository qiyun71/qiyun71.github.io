---
title: PointCloud Review
date: 2023-10-24 11:58:40
tags:
  - PointCloud
  - Registration
categories: HumanBodyReconstruction/PointCloud
---

PointCloud

- **Registration**
- **Surface Reconstruction**

Follow

- [Yochengliu/awesome-point-cloud-analysis: A list of papers and datasets about point cloud analysis (processing) (github.com)](https://github.com/Yochengliu/awesome-point-cloud-analysis)
- [zhulf0804/3D-PointCloud: Papers and Datasets about Point Cloud. (github.com)](https://github.com/zhulf0804/3D-PointCloud)
- [XuyangBai/awesome-point-cloud-registration: A curated list of point cloud registration. (github.com)](https://github.com/XuyangBai/awesome-point-cloud-registration?tab=readme-ov-file#traditional)

<!-- more -->

# Point Cloud Registration

## Review

- [A comprehensive survey on point cloud registration. (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4545266789335588865&noteId=2010010466642902272)
- [深度学习刚性点云配准前沿进展 (cjig.cn)](http://www.cjig.cn/html/2022/2/20220203.htm)
- [点云拼接注册 (geometryhub.net)](http://geometryhub.net/notes/registration)

### Old Method Library：

- Probreg is a library that implements point cloud **reg**istration algorithms with **prob**ablistic model.
- PCL: Point Cloud Library

> [neka-nat/probreg: Python package for point cloud registration using probabilistic model (Coherent Point Drift, GMMReg, SVR, GMMTree, FilterReg, Bayesian CPD) (github.com)](https://github.com/neka-nat/probreg) > [Point Cloud Library | The Point Cloud Library (PCL) is a standalone, large scale, open project for 2D/3D image and point cloud processing. (pointclouds.org)](https://pointclouds.org/)

## Greedy Grid Search

[DavidBoja/greedy-grid-search: [BMVC 2022 workshop] Greedy Grid Search: A 3D Registration Baseline (github.com)](https://github.com/davidboja/greedy-grid-search)
[DavidBoja/FAUST-partial: [BMVC 2022 workshop] 3D registration benchmark dataset FAUST-partial (github.com)](https://github.com/DavidBoja/FAUST-partial)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231019094915.png)

## SGHR

[WHU-USI3DV/SGHR: [CVPR 2023] Robust Multiview Point Cloud Registration with Reliable Pose Graph Initialization and History Reweighting (github.com)](https://github.com/WHU-USI3DV/SGHR)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231017210035.png)

## GeoTransformer

[qinzheng93/GeoTransformer: [CVPR2022] Geometric Transformer for Fast and Robust Point Cloud Registration (github.com)](https://github.com/qinzheng93/GeoTransformer?tab=readme-ov-file)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231018113631.png)

# PointCloud Surface Reconstruction

[MeshSet — PyMeshLab documentation](https://pymeshlab.readthedocs.io/en/2022.2/classes/meshset.html)
Use pymeshlab to **screened Poisson surface construction**

## NKSR

[nv-tlabs/NKSR: [CVPR 2023 Highlight] Neural Kernel Surface Reconstruction (github.com)](https://github.com/nv-tlabs/NKSR)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231017200157.png)

## FlexiCubes

[Flexible Isosurface Extraction for Gradient-Based Mesh Optimization (FlexiCubes) (nvidia.com)](https://research.nvidia.com/labs/toronto-ai/flexicubes/)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023164911.png)
