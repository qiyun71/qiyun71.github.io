---
title: An improved interval model updating method via adaptive Kriging models
date: 2024-04-09 16:18:46
tags:
  - 
categories: ModelUpdating/Interval Model Updating
---

| Title     | An improved interval model updating method via adaptive Kriging models                                                                                                     |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Sha Wei; Yifeng Chen; Hu Ding; Liqun Chen                                                                                                                                  |
| Conf/Jour | Applied Mathematics and Mechanics                                                                                                                                          |
| Year      | 2024                                                                                                                                                                       |
| Project   | [An improved interval model updating method via adaptive Kriging models \| Applied Mathematics and Mechanics](https://link.springer.com/article/10.1007/s10483-024-3093-7) |
| Paper     | https://link.springer.com/content/pdf/10.1007/s10483-024-3093-7.pdf                                                                                                        |
|           |                                                                                                                                                                            |

<!-- more -->

# AIR

## Abstract

区间模型更新（IMU）方法对样本数据的要求较低，因此在不确定模型更新中得到了广泛应用。然而，IMU 方法中的代用模型大多采用一次性构建方法。这使得代用模型的精度高度依赖于用户的经验，影响了 IMU 方法的精度。因此，本文提出了一种通过自适应克里金模型改进 IMU 方法。该方法通过通用灰色数将 IMU 问题的目标函数转化为关于上界和区间直径的两个确定性全局优化问题。这些优化问题通过自适应克里金模型和粒子群优化（PSO）方法来量化不确定参数，并完成 IMU 的优化。在构建这些自适应克里金模型的过程中，会根据灵敏度信息对样本空间进行网格化处理。然后根据最大均方误差（MMSE）准则在关键子空间进行局部采样。区间划分系数和随机抽样系数在没有人为干扰的情况下进行自适应调整，直到模型满足精度要求。通过一个**三自由度质量弹簧系统**的数值示例和一个**对接圆柱形壳体**的实验示例，证明了所提方法的有效性。 结果表明，区间模型的更新结果与实验结果十分吻合。

## Introduction

过去几十年来，有限元模型更新方法受到广泛关注[1]。经典的有限元模型更新程序主要采用确定性方法，如基于灵敏度分析的方法[2-4] 和基于代用模型的方法[5-7]。Friswell和Mottershead[8]、Ereiz 等人[9]以及 Sehgal 和 Kumar[10]对此进行了深入评述。事实上，工程应用不可避免地伴随着不确定性，如制造公差、测量噪声和材料退化造成的不确定性，以及边界条件难以准确参数化[11]。传统的模型更新方法无法合理地描述这些不确定性。随着人们对不确定性分析的日益重视，模型更新的研究逐渐从确定性方法扩展到基于不确定性的方法

模型更新中的不确定性可分为可知的不确定性和认识的不确定性[12-14]。处理已知不确定性的方法有多种，包括随机模型更新法[15-17]、fuzzy模糊模型更新法[18-19]和区间模型更新（IMU）法[20-22]。由于对样本信息的要求较少，IMU 方法逐渐被应用于机械和结构工程中的不确定性处理。IMU 方法可分为基于区间算法的方法[23-24]、顶点求解方法[25-26]和基于全局优化的方法[27-28]。由于基于区间算法和顶点解法的方法存在局限性，基于全局优化的方法受到广泛关注。Li 等人[29] 采用一阶泰勒展开表示结构的特征矩阵。根据已建立的有限元模型，用关于区间中点和区间半径的两个确定性优化问题替代 IMU 问题。Deng 等人[30] 在估计区间中点和区间半径时使用了基于径向基函数的神经网络来替代复杂的有限元模型。

作为一类很有前途的 IMU 方法，基于全局优化的方法也有其不足之处。这些方法大多采用关于区间中点和区间半径的两个确定性优化问题来替代 IMU 问题。然而，Zheng 等人[31] 发现，当系统的输入输出关系为非线性和单调关系时，这些方法不再适用于大区间问题。因此，Zheng 等人[31] 引入了通用灰色数原理，**将目标函数转化为关于上限和区间直径的两个确定性优化问题的函数**。在这项研究中，非线性单调问题得到了令人满意的更新结果。同时，在基于全局优化的方法中，代用模型的构建尤其值得注意。如果代用模型选择不当，会降低计算精度。模型更新中常用的代用模型包括响应面模型[32]、支持向量机模型[33]、径向基函数模型[34]、多元自适应回归样条线[35]、人工神经网络[36]和克里金模型[37]。上述基于全局优化的方法在构建这些代用模型时大多采用一次性构建法，即采用实验设计法一次性生成所有样本点。 这表明样本点的数量主要取决于分析人员的经验。样本点数量不足可能会导致模型精度不高。然而，当样本数量过多时，不仅会增加计算负担，还会导致过拟合问题。同时，样本点的分布具有随机性，可能导致样本分布不均匀。即使模型的精度达到了要求，也很难保证某些区域的精度。虽然已经提出了一些改进方法来提高代理模型的性能，但在 IMU 中实施这些先进技术的探索相对较少。

为了解决上述问题，本研究提出了一种通过自适应克里金模型改进的 IMU 方法。通过引入通用灰色数学，将 IMU 问题的目标函数转化为待更新参数的上界和区间直径的目标函数。然后，提出了一种自适应克里金模型构建方法。最后，利用自适应克里金模型结合粒子群优化算法（PSO）[38-39] 求解目标函数，从而确定待更新参数的区间范围。该方法建立的克里金模型可以通过自迭代达到精度要求，而无需手动设置样本数。由于样本空间被细分为多个子空间，因此在每个子空间内都会进行自适应采样，以避免样本在某些区域分布不均。这种方法可以**用较少的样本点和无需人工干预的方式提高代理模型的精度，从而提高 IMU 的精度**。

算例：
- Numerical validation of a three-degree-of-freedom mass-spring system
  - Well-separated modes
  - Closed modes
- Experimental validation of butted cylindrical shell structure
