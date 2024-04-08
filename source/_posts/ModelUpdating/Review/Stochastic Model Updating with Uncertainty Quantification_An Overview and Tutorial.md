---
title: Stochastic Model Updating with Uncertainty Quantification_An Overview and Tutorial
date: 2024-03-12 16:57:09
tags:
  - 
categories: ModelUpdating/Review
---

| Title     | Stochastic Model Updating with Uncertainty Quantification_An Overview and Tutorial                                                                                                  |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | S. Bi; Michael Beer; Scott Cogan; John Mottershead                                                                                                                                  |
| Conf/Jour | Mechanical systems and signal processing                                                                                                                                            |
| Year      | 2023                                                                                                                                                                                |
| Project   | https://www.sciencedirect.com/science/article/pii/S0888327023006921                                                                                                                 |
| Paper     | [Stochastic Model Updating with Uncertainty Quantification: An Overview and Tutorial](https://readpaper.com/pdf-annotate/note?pdfId=2201469630320002560&noteId=2201470551774580736) |

<!-- more -->

本文概述了随机模型更新的理论框架，包括模型参数化、敏感性分析、代理建模、试验分析相关性、参数校准等关键方面。**特别关注不确定性分析，将模型更新从确定性域扩展到随机域**。不确定性量化度量极大地促进了这种扩展，不再将模型参数描述为未知但固定的常数，而是具有不确定分布的随机变量，即不精确概率。因此，随机模型更新的目标不再是对单个实验进行最大保真度的单个模型预测，而是将多个实验数据的完全离散性包裹起来，减少模拟的不确定性空间。这种不精确概率的量化需要一个专门的不确定性传播过程来研究如何通过模型将输入的不确定性空间传播到输出的不确定性空间。本教程将详细**介绍前向不确定性传播和反向参数校准这两个关键方面**，以及p盒传播、基于距离的统计度量、马尔可夫链蒙特卡罗采样和贝叶斯更新等关键技术。通过解决2014年**NASA多学科UQ挑战**演示了整体技术框架，目的是鼓励读者在本教程之后复制结果。第二次实际演示是在一个新设计的基准测试台上进行的，在那里制造了**一系列具有不同几何尺寸的实验室规模的飞机模型**，遵循预先定义的概率分布，并根据其固有频率和模型形状进行测试。这样的测量数据库自然不仅包含测量误差，更重要的是包含来自结构几何预定义分布的可控不确定性。最后，讨论了开放性问题，以实现本教程的动机，为研究人员，特别是初学者，提供了从不确定性处理角度更新随机模型的进一步方向。

算例：
- NASA UQ挑战2014
- 飞机模型
