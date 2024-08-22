---
title: Frequency response function-based model updating using Kriging model
date: 2024-03-12 16:42:12
tags:
  - 
categories: ModelUpdating/Stochastic Model Updating
---

| Title     | Frequency response function-based model updating using Kriging model    |
| --------- | ----------------------------------------------------------------------- |
| Author    | Jian Wang; Chong Wang; Junpeng Zhao                                     |
| Conf/Jour | Mechanical Systems and Signal Processing                                |
| Year      | 2017                                                                    |
| Project   | https://www.sciencedirect.com/science/article/abs/pii/S0888327016304356 |
| Paper     | https://sci-hub.ee/https://doi.org/10.1016/j.ymssp.2016.10.023          |

<!-- more -->


提出了一种基于加速度频响函数(FRF)的模型更新方法，该方法**将Kriging模型作为元模型引入到优化过程**中，而不是直接迭代有限元分析。Kriging模型是一种快速运行的模型，可以减少求解时间，便于智能算法在模型更新中的应用。Kriging模型的训练样本由实验设计(DOE)生成，其响应对应于选定频率点上实验加速度频响与有限元模型对应频响的差值。考虑边界条件，**提出了一种减少训练样本数量的两步DOE方法**。第一步是从边界条件中选择设计变量，选择的变量将传递到第二步生成训练样本。将设计变量的优化结果作为设计变量的更新值对有限元进行标定，使解析频响与实验频响趋于一致。该方法在**蜂窝夹层梁复合材料结构**上进行了成功的实验，在模型更新后，分析加速度频域有了明显的改善，特别是在调整阻尼比后，分析加速度频域与实验数据吻合较好。

算例：
- experimental honeycomb beam