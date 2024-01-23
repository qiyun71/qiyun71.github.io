---
title: NeuSurf
date: 2023-12-31 22:24:49
tags:
  - 
categories: 3DReconstruction/Multi-view/Implicit Function/NeRF-based/Sparse
---

| Title     | NeuSurf: On-Surface Priors for Neural Surface Reconstruction from Sparse Input Views                                                                                                                                                                                       |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Han Huang1,2, Yulun Wu1,2, Junsheng Zhou1,2, Ge Gao1,2*, Ming Gu1,2, Yu-Shen Liu2                                                                                                                                                                                                          |
| Conf/Jour | arXiv                                                                                                                                                                                                          |
| Year      | 2023                                                                                                                                                                                                          |
| Project   |                                                                                                                                                                                                           |
| Paper     | [NeuSurf: On-Surface Priors for Neural Surface Reconstruction from Sparse Input Views (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4839146633492955137&noteId=2119047229104843264) |

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240117182026.png)

<!-- more -->

# AIR

近年来，神经隐式函数在多视图重建领域取得了显著的成果。然而，大多数现有方法都是为密集视图量身定制的，在处理稀疏视图时表现出令人不满意的性能。为了解决稀疏视图重建问题，**已经提出了几种最新的泛化隐式重建方法，但它们仍然存在训练成本高，并且仅在精心选择的视角下有效的问题**。在本文中，我们提出了一种新的稀疏视图重建框架，利用表面先验来实现高度忠实的表面重建。具体来说，我们设计了几个全局几何对齐和局部几何细化约束，以共同优化粗形状和精细细节。为了实现这一点，我们训练了一个神经网络，从SfM获得的表面点中学习全局隐式域，然后将其作为粗几何约束。为了利用局部几何一致性，我们将表面上的点投影到可见和不可见的视图上，将投影特征的一致损失视为精细的几何约束。在两种流行的稀疏设置下，DTU和BlendedMVS数据集的实验结果表明，与最先进的方法相比，有显著的改进。