---
title: SGHR
date: 2023-10-17 21:49:14
tags:
  - PointCloud
  - Registration
categories: HumanBodyReconstruction/PointCloud
---

| Title     | Robust Multiview Point Cloud Registration with Reliable Pose Graph Initialization and History Reweighting                                                                                                                 |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Haiping Wang and Yuan Liu and Zhen Dong and Yulan Guo and Yu-Shen Liu and Wenping Wang and Bisheng Yang                                                                                                                   |
| Conf/Jour | CVPR                                                                                                                                                                                                                      |
| Year      | 2023                                                                                                                                                                                                                      |
| Project   | [WHU-USI3DV/SGHR: [CVPR 2023] Robust Multiview Point Cloud Registration with Reliable Pose Graph Initialization and History Reweighting (github.com)](https://github.com/WHU-USI3DV/SGHR?tab=readme-ov-file)              |
| Paper     | [Robust Multiview Point Cloud Registration with Reliable Pose Graph Initialization and History Reweighting (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4740850412790218753&noteId=2008923607452724224) |

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231017210035.png)

<!-- more -->

# Abstract

本文提出了一种点云多视图配准的新方法。以前的多视图配准方法依赖于穷举成对配准构造密集连接的位姿图，并在位姿图上应用迭代重加权最小二乘(IRLS)计算扫描位姿。然而，构造一个密集连接的图是耗时的，并且包含大量的离群边，这使得后续的IRLS很难找到正确的姿势。为了解决上述问题，我们首先提出使用神经网络来估计扫描对之间的重叠，这使我们能够构建一个稀疏但可靠的姿态图。然后，我们在IRLS方案中设计了一种新的历史重加权函数，该函数对图上的离群边具有较强的鲁棒性。与现有的多视图配准方法相比，我们的方法在3DMatch数据集上的配准召回率提高了11%，在ScanNet数据集上的配准误差降低了13%，同时所需的成对配准减少了70%。进行了全面的ablation研究，以证明我们设计的有效性

传统点云配准：
- **首先**，采用成对配准算法[28,46,49]，穷尽估计所有N2扫描对的相对姿态，形成一个完全连通的姿态图。图的边表示扫描对的相对位置，节点表示扫描。
- 由于密集姿态图可能包含两次不相关扫描之间不准确甚至不正确的相对姿态(异常值)，因此在**第二阶段**，通过加强周期一致性[30]来联合优化这些成对姿态，以拒绝异常边并提高精度。对于第二阶段，最新的方法，包括手工方法[5,13,29]或基于学习的方法[21,30,55]，采用迭代加权最小二乘(IRLS)方案。Iteratively Reweighted Least Square (IRLS)

# Method

(1)给定N个未对齐的部分扫描，我们的目标是将所有这些扫描注册到(4)一个完整的点云中。我们的方法有两个贡献
- (2)学习全局特征向量初始化稀疏姿态图，使离群点更少，减少了两两配准所需的次数。
- (3)提出了一种新的IRLS方案。在我们的IRLS方案中，我们从全局特征和两两配准中初始化权重。然后，我们设计了一个历史加权函数来迭代地改进姿态，提高了对异常值的鲁棒性。

![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231017215645.png)


# Experiments

我们在三个广泛使用的基准上评估我们的方法:3DMatch/3DLoMatch数据集[28,59]，ScanNet数据集[16]和ETH数据集[44]