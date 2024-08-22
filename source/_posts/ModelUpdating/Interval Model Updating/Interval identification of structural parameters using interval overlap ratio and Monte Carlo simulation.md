---
title: Interval identification of structural parameters using interval overlap ratio and Monte Carlo simulation
date: 2024-03-12 16:47:06
tags:
  - 
categories: ModelUpdating/Interval Model Updating
---

| Title     | Interval identification of structural parameters using interval overlap ratio and Monte Carlo simulation                                                                                                 |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Deng Zhongmin a , Guo Zhaopu a,b 北航                                                                                                                                                                      |
| Conf/Jour | Advances in Engineering Software                                                                                                                                                                         |
| Year      | 2018                                                                                                                                                                                                     |
| Project   | https://www.sciencedirect.com/science/article/abs/pii/S096599781731164X                                                                                                                                  |
| Paper     | [Interval identification of structural parameters using interval overlap ratio and Monte Carlo simulation](https://readpaper.com/pdf-annotate/note?pdfId=2201610607974204928&noteId=2201611039416835840) |

<!-- more -->

# AIR

## Abstract

从不确定性传播和不确定性量化的角度，**提出了一种新的区间有限元模型更新策略**，用于结构参数的区间识别。将蒙特卡罗仿真与surrogate模型相结合，可以有效地获得系统响应的精确区间估计。利用区间长度的概念，构造了一种新的定量指标——**区间重叠率(IOR)**，用来表征分析数据与实测数据区间分布的一致性。构造并解决了两个不确定结构参数标称值和区间半径估计的优化问题。最后，**通过数值和实验**验证了该方法在结构参数区间识别中的可行性。

## Introduction

在过去的几十年里，特别是近年来，人们对利用实验数据作为精确参考数据来改进有限元预测的兴趣日益浓厚，这被称为有限元模型更新[1]。有限元模型更新是一个反问题，用以识别和调整建模参数，从而更好地预测实际结构[2]的响应行为。确定性模型更新方法[3–5]现在已经众所周知，并广泛应用于工业规模的结构。在这些方法中，每个更新参数都被认为是通过最小化计算结果和来自单个物理结构[6]的测量数据之间的误差来识别的。然而，与几何尺寸和材料性能相关的参数不确定性存在于大多数现实工程结构中。因此，一个涉及不确定性分析的模型更新过程是非常重要的，因此模型更新的目的就成为了对参数[7,8]的范围或分布的估计。
随机模型更新方法[9-19]是为处理几何尺寸的制造公差、材料属性的离散性等引起的参数变化而开发的。Fonseca 等人[11]提出了一种基于最大似然函数的更新算法。基于 MC 的模型更新程序[9,10,12,13,18]在识别参数变异性方面相对容易实现，但也需要大量的计算费用。同时，扰动方法[15,16] 已成功应用于随机模型更新，其中结构参数和测量响应是确定性部分和随机变化的总和。
然而，充分的概率估计总是需要足够的测量信息，而这在工程实践中往往是不切实际的 [8,20]。在这种情况下，区间法[21]被引入作为量化不确定参数的一种有用替代方法。在用区间数描述参数不确定性的区间模型更新领域，已经有人尝试[8,22,23]解决区间逆问题。为了解决区间模型更新问题，Khodaparast 等人[8]首先提出了参数顶点法，该方法仅对 FE 模型的特定参数化有效。由于这一缺陷，他们随后提出了基于Kriging预测器灵敏度分析的一般情况下的全局优化方法。Fang 等人[22]针对区间模型更新问题提出了区间响应面模型（IRSM）。应用 IRSM 可以简化区间逆问题的建立和实现，因为该模型有助于在区间模型更新过程中直接执行区间算法。需要注意的是，IRSM 是在二阶多项式模型的基础上构建的，无法考虑更新参数之间的交互项。此外，通过使用扰动技术，Deng 等人[23] 开发了两步法，用于更新不确定参数的平均值和区间半径。

> 8[Interval model updating with irreducible uncertainty using the Kriging predictor - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327010003286)
> 22[An interval model updating strategy using interval response surface models - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327015000229)
> 23[Interval model updating using perturbation method and Radial Basis Function neural networks - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327016303417)

在区间模型更新的研究和开发过程中，不确定性分析日益成为最关键的技术之一。不确定性分析问题一般可分为两个问题：不确定性传播和不确定性量化。目前，不确定性传播的区间方法主要有区间算术法(interval arithmetic)[24]、扰动法[25]、顶点法[8]和 MC 仿真法。区间运算法包括一组运算，由于忽略了运算元之间的相关性，往往会高估系统响应的范围。顶点法仅适用于输入和输出之间存在单调关系的特定情况。虽然扰动法考虑到了计算效率，但它取决于区间变量的初始值和不确定水平。由于精度高、过程简单，MC 仿真被认为是一种适用于输入/输出变量不确定性传播的技术。然而，庞大的计算量阻碍了直接 MC 仿真的实施。另一方面，不确定性量化的目的是提供一个定量指标，表征分析预测与测量观测之间的区间范围的一致性。迄今为止，常用的区间模型更新定量指标都是基于区间边界 [8,22]、区间半径 [23] 或区间阈值 [24] 来构建的。

> 24[Analysis of Uncertain Structural Systems Using Interval Analysis | AIAA Journal](https://arc.aiaa.org/doi/abs/10.2514/2.164)
> 25[Modified perturbation method for eigenvalues of structure with interval parameters | Science China Physics, Mechanics & Astronomy](https://link.springer.com/article/10.1007/s11433-013-5328-6)
> 

如上所述，本文在不确定性传播和不确定性量化方面提出了一种新的区间模型更新策略。首先，选择不确定参数和区间分布，并通过 MC 分析在模型中进行前向传播，以提供系统响应的精确区间范围。同时，为了减少 MC 仿真的计算量，在更新参数区间的过程中使用了代用模型。其次，利用区间长度的概念，构建了名为 IOR 的新型定量指标来表征区间分布的一致性。第三，构建并求解了两个优化方程，用于估算不确定结构参数的标称值和区间半径。最后，给出了数值和实验案例研究，以说明所提方法在结构参数区间识别中的可行性。

不确定性分析问题一般可分为两个方面:不确定性传播和不确定性量化
- 不确定性传播的主要区间方法有区间算法[24]、摄动法[25]、顶点法[8]和MC模拟
  - 各种方法的缺陷：由于忽略了操作数之间的相关性，包含一组操作的区间算法往往高估了系统响应的范围。顶点方法仅对输入和输出之间存在单调关系的特殊情况有效。摄动法虽然考虑了计算效率，但它依赖于区间变量的初值和不确定水平。MC仿真由于精度高、过程简单，被认为是求解输入/输出变量不确定性传播的一种合适的方法。然而，庞大的计算量阻碍了直接MC模拟的实现。
- 不确定度量化的目的是提供一种定量度量，以表征分析预测和测量观测之间的区间范围的一致性。到目前为止，常用的区间模型更新定量指标是基于区间界[8,22]、区间半径[23]或区间交集[26,27]来构建的

算例：
- Numerical case studies: **a mass-spring system**具有良好分离和紧密模态的三自由度质量-弹簧系统
  - well-separated modes
  - close modes
- Experimental case study: **steel plate structures**