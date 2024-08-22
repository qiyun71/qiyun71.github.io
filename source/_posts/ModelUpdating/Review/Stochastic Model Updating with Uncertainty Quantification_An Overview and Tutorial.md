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

# Theory

Uncertainty
- Epistemic Uncertainty
  - 源自认知信息不足，可以减少乃至消除
- Aleatory Uncertainty
  - 源自系统/结构固有的随机性

根据参数中是否存在Epistemic Uncertainty/Aleatory Uncertainty，将参数分为四类
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240412091330.png)

- Category 1：具有完全确定值的常数
- Category 2：未知但固定的常数(区间)
- Category 3：具有完全确定分布性质的随机变量(分布格式、均值和方差)，称为精确概率
- Category 4：随机变量，分布性质尚未完全确定，称为不精确概率，可以通过P-box来建模，P-box中无限数量的CDF曲线构成了概率空间中的特定区域

确定模型修正适用于Category 2参数，确定预定义区间内的特定值
随机模型修正适用于Category 2~4类参数

## 随机模型修正

随机模型修正是一个综合技术系统，其关键方面包括灵敏度分析、测试分析相关性和参数校准等，其中不确定性分析作为关键角色被深度整合，以促进模型修正从确定性领域向随机领域发展

模型修正涵盖模型准备、振动实验、灵敏度分析、误差定位、参数校准等方面。
基于灵敏度的方法对典型的模态特征有效，例如自然频率和模态形状，其灵敏度可以从理论上从刚度和质量推导出来。这些理论推导对于现代结构工程来说是不切实际的，因为大规模结构系统呈现出强大的非线性动力学或瞬态分析
蒙特卡罗方法，通过多个确定性模型评估，在模型参数和任何输出特征之间提供直接联系，且随机抽样过程具有不确定性分析的自然适应性，因为样本总是从概率假设中获得的。计算技术的重大发展进一步促进了这一趋势，这使得大尺寸抽样成为可能，从而可以精确估计变量的统计信息

**随机模型修正主要可以分为频率派和贝叶斯派**
- 频率主义方法侧重于优化技术，以最小化现有测量和模型模拟之间的差异。
- 贝叶斯方法自然涉及更新中的不确定性处理。它还具有在存在非常罕见的测量数据的情况下捕获不确定性信息的优越性。因此,贝叶斯方法已被采用为随机模型更新中最受欢迎的技术之一
- 此外不精确概率技术也具有相当大的潜力，如
  - interval probabilities
  - evidence theory
  - info-gap theory
  - fuzzy probabilities

随机模型修正包含几个关键步骤：feature definition, model parameterisation, surrogate modelling, parameter calibration, and model validation.
- Feature definition：有限元模型/实验输出的特征，可以是模态量(natural frequencies and mode shapes)，也可以是连续的量(例如时域系统响应或频率响应函数)。不同的输出特征比较需要不同的UQ指标
- Parameterisation and sensitivity analysis：参数化总是基于工程判断进行，参数化会产生大量可能对模型输出不确定或有意义的参数。随后灵敏度分析被开发为一种典型的技术，用于测量输入参数相对于输出特征的显着性，从而帮助选择下一步要校准的关键参数
- Surrogate modelling：代理有限元模型进行不确定性传播。The polynomial function，radial basis function，support vector machine，Kriging function，and neural network(BP or others)
- Parameter calibration：本质上是一个反向过程，以模拟和测量输出特征之间的差异为参考，并关注如何校准输入参数的原理和技术
  - 优化算法(eg: SSA)：将此任务描述为优化问题，输出差异被用来构建目标函数，通过搜索合适的参数值及其认识空间作为优化约束，该目标函数将被最小化(*基于灵敏度的优化方法*)
  - Bayes：其中参数的先验分布预计会根据现有测量的似然函数进行更新，并且更新的后验分布以更小的认识不确定性获得（*贝叶斯方法*）
- Model validation：模型验证是评估模型预测能力的重要步骤


## UQ
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240412103257.png)

- 欧氏距离$d_E\big(\mathbf{Y}_{exp},\mathbf{Y}_{sim}\big)=\sqrt{\big(\overline{\mathbf{Y}_{exp}}-\overline{\mathbf{Y}_{sim}}\big)\big(\overline{\mathbf{Y}_{exp}}-\overline{\mathbf{Y}_{sim}}\big)^T}$
- 马氏距离$d_M\left(\mathbf{Y}_{exp},\mathbf{Y}_{sim}\right)=\sqrt{\left(\overline{\mathbf{Y}_{exp}}-\overline{\mathbf{Y}_{sim}}\right)\mathbf{C}^{-1}\left(\overline{\mathbf{Y}_{exp}}-\overline{\mathbf{Y}_{sim}}\right)^T}$
  - C is the “pooled” covariance matrix of both the simulation and measurement samples：$\mathbf{C}=\frac{(m-1)\mathbf{C}_{sim}+(n-1)\mathbf{C}_{exp}}{m+n-2}$
  - m and n are the numbers of simulated and measured sample
- 巴氏距离$d_B\big(\mathbf{Y}_{exp},\mathbf{Y}_{sim}\big)=-\log\bigg[\int_y\sqrt{P_{exp}(y)P_{sim}(y)}\mathrm{d}y\bigg]$


> [how to calculate mahalanobis distance in pytorch? - Stack Overflow](https://stackoverflow.com/questions/65328887/how-to-calculate-mahalanobis-distance-in-pytorch)
> [概率分布之间的距离度量以及python实现 - 咖啡陪你 - 博客园](https://www.cnblogs.com/ltkekeli1229/p/15751619.html)
> [机器学习中不同的距离定义汇总（四：概率分布之间的距离）](https://blog.zhenyuan.cool/p/99b411b1823d6d8a/)

```
# 马氏距离
def mahalanobis(u, v, cov):
delta = u - v
m = torch.dot(delta, torch.matmul(torch.inverse(cov), delta))
return torch.sqrt(m)

# 巴氏距离
import numpy as np
p=np.asarray([0.65,0.25,0.07,0.03])
q=np.array([0.6,0.25,0.1,0.05])
BC=np.sum(np.sqrt(p*q))

#Hellinger距离：
h=np.sqrt(1-BC)
#巴氏距离：
b=-np.log(BC)
```