---
title: Interval model updating with irreducible uncertainty using the Kriging predictor
date: 2024-03-13 15:29:04
tags:
  - 
categories: ModelUpdating/Interval Model Updating
---

| Title     | Interval model updating with irreducible uncertainty using the Kriging predictor |
| --------- | -------------------------------------------------------------------------------- |
| Author    | H. H. Khodaparast; J. Mottershead; K. Badcock                                    |
| Conf/Jour | Mechanical Systems and Signal Processing                                         |
| Year      | 2011                                                                             |
| Project   | https://www.sciencedirect.com/science/article/abs/pii/S0888327010003286          |
| Paper     | https://sci-hub.ee/https://doi.org/10.1016/j.ymssp.2010.10.009                   |

<!-- more -->

定义了存在不可还原的不确定测量数据时的区间模型更新，并提供了两种情况下的解决方案。在第一种情况下，**使用参数顶点解决方案**，但发现仅对有限元模型的特定参数化和特定输出数据有效。在第二种情况下，考虑**使用元模型作为完整有限元数学模型的代用模型**，从而获得通用解决方案。因此，输入数据区域与通过回归分析获得参数的输出数据区域进行映射。本文选择**Kriging预测器**作为元模型，发现它能够非常准确地预测输入和输出参数变化的区域。基于**Kriging预测器**制定了区间模型更新方法，并开发了迭代程序。该方法使用一个具有良好分离模式和接近模式的**三自由度质量弹簧系统**进行了数值验证。Kriging插值法的一个显著优点是，它可以使用传统的有限元模型修正方法难以使用的更新参数。在一个实验练习中，选择了框架结构中**两根梁的位置**作为更新参数，这就是一个例子。