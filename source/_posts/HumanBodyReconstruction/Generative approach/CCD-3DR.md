---
title: CCD-3DR
date: 2023-11-07 17:25:01
tags:
  - 3DReconstruction
categories: HumanBodyReconstruction/Generative approach
---

| Title     | CCD-3DR: Consistent Conditioning in Diffusion for Single-Image 3D Reconstruction                                                                                                                 |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Author    | Yan Di1, Chenyangguang Zhang2, Pengyuan Wang1, Guangyao Zhai1, Ruida Zhang2, Fabian Manhardt3, Benjamin Busam1, Xiangyang Ji2, and Federico Tombari1,3                                           |
| Conf/Jour | arXiv                                                                                                                                                                                            |
| Year      | 2023                                                                                                                                                                                                 |
| Project   |                                                                                                                                                                                                  |
| Paper     | [CCD-3DR: Consistent Conditioning in Diffusion for Single-Image 3D Reconstruction (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4789424767274844161&noteId=2014146864821066240) |

No Code
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021115606.png)

<!-- more -->

# Abstract

本文提出了一种利用扩散模型对**单幅RGB图像**中捕获的物体生成三维稀疏点云的形状重建方法。最近的方法通常利用全局嵌入或基于局部投影的特征作为指导扩散模型的条件。然而，这种策略不能将去噪点云与给定图像一致对齐，导致条件不稳定，性能较差。在本文中，我们提出了CCD-3DR，它利用了一种新的中心扩散概率模型来进行一致的局部特征条件反射。我们将扩散模型中的噪声和采样点云约束到一个子空间中，在这个子空间中，点云中心在正向扩散过程和反向扩散过程中保持不变。稳定的点云中心进一步充当锚，将每个点与其相应的基于投影的局部特征对齐。在合成基准ShapeNet-R2N2上进行的大量实验表明，CCD-3DR的性能大大优于所有竞争对手，改进幅度超过40%。我们还提供了实际数据集Pix3D的结果，以彻底展示CCD3DR在实际应用中的潜力。代码将很快发布


