---
title: Bayesian inversion for imprecise probabilistic models using a novel entropy-based uncertainty quantification metric
date: 2024-03-12 16:49:50
tags:
  - 
categories: ModelUpdating/Interval Model Updating
---

| Title     | Bayesian inversion for imprecise probabilistic models using a novel entropy-based uncertainty quantification metric |
| --------- | ------------------------------------------------------------------------------------------------------------------- |
| Author    | Lechang Yang; S. Bi; M. Faes; M. Broggi; M. Beer                                                                    |
| Conf/Jour | Mechanical Systems and Signal Processing                                                                            |
| Year      | 2022                                                                                                                |
| Project   | https://www.sciencedirect.com/science/article/abs/pii/S0888327021003496                                             |
| Paper     | https://sci-hub.ee/https://doi.org/10.1016/j.ymssp.2021.107954                                                      |

<!-- more -->

不确定性量化度量在动态系统反问题中具有关键地位，因为它们量化了数值预测样本与收集到的观测值之间的差异。这种度量通过奖励这种差异规范较小的样本和惩罚其他样本来发挥其作用。在本文中，我们利用Jensen-Shannon散度提出了一个**新的基于熵的度量**。与其他现有的基于距离的度量相比，一些独特的性质使这种基于熵的度量成为解决混合不确定性(即，不确定性和认知不确定性的组合)存在的逆问题的有效工具，例如在不精确概率的背景下遇到的逆问题。在实现方面，**开发了一种近似贝叶斯计算方法**，其中所提议的度量被完全嵌入。为了减少计算量，采用离散化的分簇算法代替传统的多变量核密度估计。为了验证目的，首先演示了一个静态案例研究，其中与其他三种行之有效的方法进行了比较。为了突出其在复杂动态系统中的潜力，我们将我们的方法应用于**2014年NASA LaRC不确定性量化挑战问题**，并将所获得的结果与文献中其他6个研究小组的结果进行了比较。这些例子说明了我们的方法在静态和动态系统中的有效性，并显示了它在实际工程案例中的前景，如结构健康监测与动态分析相结合。

算例：
- Simply supported beam简支梁
- Practical application example: NASA UQ challenge problem2014