---
title: SG-GAN
date: 2023-10-21 19:42:03
tags:
  - GAN
categories: 3DReconstruction/Single-view
---

| Title     | SG-GAN: Fine Stereoscopic-Aware Generation for 3D Brain Point Cloud Up-sampling from a Single Image                                                                                                                                                                                                              |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Bowen Hu, Baiying Lei, Shuqiang Wang, Senior Member, IEEE                                                                                                                                                                                                                    |
| Conf/Jour | arXiv                                                                                                                                                                                                                    |
| Year      | 2023                                                                                                                                                                                                                    |
| Project   |                                                                                                                                                                                                                     |
| Paper     | [SG-GAN: Fine Stereoscopic-Aware Generation for 3D Brain Point Cloud Up-sampling from a Single Image (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4762223320204574721&noteId=2014610279544785920) |

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021193539.png)

Stereoscopic-aware graph generative adversarial network (SG-GAN)

<!-- more -->

# Abstract

在间接狭窄手术环境下的微创颅脑手术中，三维脑重建是至关重要的。然而，随着一些新型微创手术(如脑机接口手术)对精度的要求越来越高，点云(PC)等传统三维重建的输出面临着样本点过于稀疏、精度不足的挑战。另一方面，高密度点云数据集的稀缺，给直接重建高密度脑点云训练模型带来了挑战。在这项工作中，提出了一种新的模型，称为立体感知图形生成对抗网络(SG-GAN)，该模型具有两个阶段，可以在单个图像上生成精细的高密度 PC。Stage-I GAN 根据给定的图像绘制器官的原始形状和基本结构，产生 Stage-I 点云。第二阶段 GAN 采用第一阶段的结果，生成具有详细特征的高密度点云。ii 阶段 GAN 能够通过上采样过程纠正缺陷并恢复感兴趣区域(ROI)的详细特征。此外，开发了一种基于无参数注意的自由变换模块来学习输入的有效特征，同时保持良好的性能。与现有方法相比，SG-GAN 模型在视觉质量、客观测量和分类性能方面表现出优异的性能，pc - pc 误差和倒角距离等多个评价指标的综合结果表明。

# Method

SG-GAN 架构包括**一个基于 ResNet 和 FTM 的编码器**，以及**两个 GAN 结构**。该方法避免了直接生成的重构误差，仅通过 PC 上采样过程修正缺陷和恢复细节。制定了统一的信息流，使相邻模块之间能够通信，并解决其输入和输出的差异。

## Fast feature aggregating encoder based on free transforming module (FTM)

Develop a free transforming module (FTM) by introducing a **parameter-free self-attention mechanism** into the conventional **ResNet network**



![image.png|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021200340.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231021200651.png)

## Loss



$\begin{aligned}\mathcal{L}_{CD}&=\sum_{y'\in Y'}min_{y\in Y}||y'-y||_2^2+\sum_{y\in Y}min_{y'\in Y'}||y-y'||_2^2,\end{aligned}$

$\mathcal{L}_{EMD}=min_{\phi:Y\rightarrow Y^{\prime}}\sum_{x\in Y}||x-\phi(x)||_{2}$ , φ indicates a parameter of bijection.


# Conclusion

本文针对脑机接口微创手术中现有三维重建方法清晰度低、密度低的问题，提出了一种新的图像到 pc 的重建网络 SG-GAN。提出的 SG-GAN 实现了3D 脑重建技术，手术导航可以从中受益。考虑到手术导航中对实时信息反馈的需求以及相关的计算时间成本，我们选择利用点云作为我们提出的模型的表示，并将单个图像作为输入。
一种简单的替代方法是训练一个基本模型，并在一个步骤中从输入重建整个高密度点云。然而，由于以下两个方面的原因，这种替代方案可能不切实际。(1)不能充分利用原始图像中的先验知识和已知信息;(2)生成的高密度点云的一些微观结构在几何角度上没有得到充分的调整，从而导致较大的细节误差。
通过集成几个互补模块，所提出的 SG-GAN 能够完成复杂脑形状的预定重建。共同完成预定的复杂重构和上采样任务。设计了一种基于数值计算的无参数注意机制来构成 GAN 的编码器。该编码器利用空间域自关注来调整提取特征的权重，同时保证手术场景的时间敏感性，从而使输出的特征向量更具形状合理性。然后，设计两个不同阶段的 gan 来形成生成网络。**第一阶段 GAN 的重点是勾勒出目标大脑的特定形状，并利用这些特征生成低密度点云。第二阶段 GAN 的重点是描绘目标大脑的具体细节，并修复第一阶段的一些生成错误，以重建最终的高密度 PC**。
目前，术中MRI技术对手术导航有了很大的改进。在本研究中，我们利用单片脑MR图像作为输入，重建相应的三维脑形状，得到的结果是有希望的。我们的目标是在脑外科手术导航中，为医生提供高密度的三维脑形态的即时访问。对于未来的工作，我们将与临床医生合作，从现实世界中收集脑外科导航产生的数据，以便我们可以尽可能地消除所提出模型的输入约束。最后，我们的模型可以适应现实世界数据集的输入，并获得相同的竞争三维形状结构。