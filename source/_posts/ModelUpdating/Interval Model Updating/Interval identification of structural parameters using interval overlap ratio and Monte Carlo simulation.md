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

从不确定性传播和不确定性量化的角度，**提出了一种新的区间有限元模型更新策略**，用于结构参数的区间识别。将蒙特卡罗仿真与surrogate模型相结合，可以有效地获得系统响应的精确区间估计。利用区间长度的概念，构造了一种新的定量指标——**区间重叠率(IOR)**，用来表征分析数据与实测数据区间分布的一致性。构造并解决了两个不确定结构参数标称值和区间半径估计的优化问题。最后，**通过数值和实验**验证了该方法在结构参数区间识别中的可行性。

不确定性分析问题一般可分为两个方面:不确定性传播和不确定性量化
- 不确定性传播的主要区间方法有区间算法[24]、摄动法[25]、顶点法[8]和MC模拟
  - 各种方法的缺陷：由于忽略了操作数之间的相关性，包含一组操作的区间算法往往高估了系统响应的范围。顶点方法仅对输入和输出之间存在单调关系的特殊情况有效。摄动法虽然考虑了计算效率，但它依赖于区间变量的初值和不确定水平。MC仿真由于精度高、过程简单，被认为是求解输入/输出变量不确定性传播的一种合适的方法。然而，庞大的计算量阻碍了直接MC模拟的实现。
- 不确定度量化的目的是提供一种定量度量，以表征分析预测和测量观测之间的区间范围的一致性。到目前为止，常用的区间模型更新定量指标是基于区间界[8,22]、区间半径[23]或区间交集[26,27]来构建的

算例：
- Numerical case studies: **a mass-spring system**具有良好分离和紧密模态的三自由度质量-弹簧系统
  - well-separated modes
  - close modes
- Experimental case study: **steel plate structures**