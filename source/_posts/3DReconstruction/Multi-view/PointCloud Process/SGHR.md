---
title: SGHR
date: 2023-10-17 21:49:14
tags:
  - PointCloud
  - Registration
categories: 3DReconstruction/Multi-view
---

| Title     | Robust Multiview Point Cloud Registration with Reliable Pose Graph Initialization and History Reweighting                                                                                                                 |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Haiping Wang and Yuan Liu and Zhen Dong and Yulan Guo and Yu-Shen Liu and Wenping Wang and Bisheng Yang                                                                                                                   |
| Conf/Jour | CVPR                                                                                                                                                                                                                      |
| Year      | 2023                                                                                                                                                                                                                      |
| Project   | [WHU-USI3DV/SGHR: [CVPR 2023] Robust Multiview Point Cloud Registration with Reliable Pose Graph Initialization and History Reweighting (github.com)](https://github.com/WHU-USI3DV/SGHR?tab=readme-ov-file)              |
| Paper     | [Robust Multiview Point Cloud Registration with Reliable Pose Graph Initialization and History Reweighting (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4740850412790218753&noteId=2008923607452724224) |

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231017210035.png)

Issue:
[How should I train my dataset? · Issue #4 · WHU-USI3DV/SGHR (github.com)](https://github.com/WHU-USI3DV/SGHR/issues/4)
I think several point clouds of a single statue is not enough for training deep descriptors. I suggest to directly use pairwise registration models such as [Geotrainsformer](https://github.com/qinzheng93/GeoTransformer) pre-trained on object-level datasets such as ModelNet40 **to solve the pairwise registrations**.  
And adopt SGHR's **transformation synchronization** section to solve the global consistent scan poses.

<!-- more -->

# Abstract

本文提出了一种点云多视图配准的新方法。以前的多视图配准方法依赖于穷举成对配准构造密集连接的位姿图，并在位姿图上应用迭代重加权最小二乘(IRLS)计算扫描位姿。然而，构造一个密集连接的图是耗时的，并且包含大量的离群边，这使得后续的IRLS很难找到正确的姿势。为了解决上述问题，我们首先提出使用神经网络来估计扫描对之间的重叠，这使我们能够构建一个稀疏但可靠的姿态图。然后，我们在IRLS方案中设计了一种新的历史重加权函数，该函数对图上的离群边具有较强的鲁棒性。与现有的多视图配准方法相比，我们的方法在3DMatch数据集上的配准召回率提高了11%，在ScanNet数据集上的配准误差降低了13%，同时所需的成对配准减少了70%。进行了全面的ablation研究，以证明我们设计的有效性

传统点云配准：
- **首先**，采用成对配准算法[28,46,49]，穷尽估计所有N2扫描对的相对姿态，形成一个完全连通的姿态图。图的边表示扫描对的相对位置，节点表示扫描。
- 由于密集姿态图可能包含两次不相关扫描之间不准确甚至不正确的相对姿态(异常值)，因此在**第二阶段**，通过加强周期一致性[30]来联合优化这些成对姿态，以拒绝异常边并提高精度。对于第二阶段，最新的方法，包括手工方法[5,13,29]或基于学习的方法[21,30,55]，采用迭代加权最小二乘(IRLS)方案。Iteratively Reweighted Least Square (IRLS)

# Method

(1)给定N个未对齐的部分扫描，我们的目标是将所有这些扫描注册到(4)一个完整的点云中。我们的方法有两个贡献
- (2)学习全局特征向量初始化稀疏姿态图，使离群点更少，减少了两两配准所需的次数。
    - **Global feature extraction**：**YOHO** for extracting local descriptors , **NetVLAD** to extract a global feature F (train with a L1 loss between the predicted overlap score and the ground-truth overlap ratio.)
    - **Sparse graph construction**：overlap score ==>For each scan, select other k scan pairs with the largest overlap scores to connect with the scan
    - estimate a relative pose on the scan pair from their extracted local descriptors(follow YOHO to apply nearest neighborhood matcher )
- (3)提出了一种新的IRLS方案。在我们的IRLS方案中，我们从全局特征和两两配准中初始化权重。然后，我们设计了一个历史加权函数来迭代地改进姿态，提高了对异常值的鲁棒性。*(IRLS的关键思想是在每条边上关联一个权值来表示每个扫描对的可靠性。这些权重被迭代地细化，使得离群边缘的权重较小，这样这些离群相对姿态就不会影响最终的全局姿态)*
    - Weight initialization
    - Pose synchronization, 给定edge weights 和 input relative poses求解global scan poses
        - Rotation synchronization, ref:  Global Motion Estimation from Point Matches
        - Translation synchronization, ref:  Learning Transformation Synchronization 最小二乘法求解
    - History reweighting function

![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231017215645.png)


# Experiments

我们在三个广泛使用的基准上评估我们的方法:3DMatch/3DLoMatch数据集[28,59]，ScanNet数据集[16]和ETH数据集[44]

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231018170845.png)
我们的全局特征提取网络架构如图A.1所示。我们采用与[49]架构相同的YOHO进行32点局部特征提取。更多的局部特征提取细节可以在[49]中找到。提取的局部特征通过NetVLAD层聚合为全局特征[3]。我们将netvlad中的集群数量设置为64，因此全局特征的维度为2048。请参考[3]了解更多的全局特征聚合细节。

我们使用预训练好的YOHO[49]进行局部特征提取，并使用3DMatch[59]训练分割中的46个场景训练N etVLAD层。我们采用以下数据扩充。对于3DMatch训练集中的每个场景，我们首先随机抽取α∈[8,60]扫描作为图节点。然后，在每次扫描中，我们随机抽取β∈[1024,5000]个关键点来提取YOHO特征。将α扫描的局部特征输入到NetVLAD中提取α扫描的全局特征。然后，我们通过穷列关联每两个全局特征来计算α2重叠分数，并计算真实重叠比率与预测重叠分数之间的L1距离作为训练损失。我们将批大小设置为1，并使用学习率为1e-3的Adam优化器。学习率每50个历元呈指数衰减0.7倍。总的来说，我们训练了netv LAD 300个epoch。

# Conclusion

本文提出了一种新的多视点云配准方法。该方法的关键是基于学习的稀疏姿态图构建，该方法可以估计两次扫描之间的重叠比，使我们能够选择高重叠的扫描对来**构建稀疏但可靠的图**。在此基础上，提出了一种新的**历史加权函数**，提高了IRLS方案对异常值的鲁棒性，并对姿态校正有较好的收敛性。所提出的方法在室内和室外数据集上都具有最先进的性能，而且配对配准的次数要少得多。