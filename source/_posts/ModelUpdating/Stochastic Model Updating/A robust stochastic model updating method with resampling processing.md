---
title: A robust stochastic model updating method with resampling processing
date: 2024-03-12 16:34:57
tags:
  - 
categories: ModelUpdating/Stochastic Model Updating
---

| Title     | A robust stochastic model updating method with resampling processing                                                                                                 |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Yanlin Zhao, Zhongmin Deng ⇑, Xinjie Zhang                                                                                                                           |
| Conf/Jour | Mechanical Systems and Signal Processing                                                                                                                             |
| Year      | February 2020                                                                                                                                                        |
| Project   | https://www.sciencedirect.com/science/article/abs/pii/S0888327019307150                                                                                              |
| Paper     | [A robust stochastic model updating method with resampling processing](https://readpaper.com/pdf-annotate/note?pdfId=2121926335166128896&noteId=2121926748338632704) |

<!-- more -->


为了更好地估计参数的不确定性，提出了一种鲁棒随机模型更新框架。在该框架中，为了提高鲁棒性，重采样过程主要设计用于处理不良样本点，特别是**有限样本量问题**。其次，**提出了基于巴塔查里亚距离和欧几里得距离的平均距离不确定度UQ度量**，以充分利用测量的可用信息。随后**采用粒子群优化算法**对结构的输入参数进行更新。最后以**质量-弹簧系统和钢板结构为例**说明了该方法的有效性和优越性。通过将测量样品加入不良样品，讨论了重采样过程的作用。

- 有限元模型具有一定的不确定性，主要是由于模型的简化、近似以及模型参数的不确定性，如弹性模量、几何尺寸、边界条件、静、动载荷条件等。
- 实验系统的测量都具有一定的不确定度。这种不确定性与难以控制的随机实验效应有关，如制造公差引入的偏差，随后信号处理期间的测量噪声或有限的测量数据。

算例：
- a mass-spring system
- steel plate structures

