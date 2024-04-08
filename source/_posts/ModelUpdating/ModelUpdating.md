---
title: Model Updating Review
date: 2023-12-22 15:03:03
tags: 
categories: ModelUpdating
---

**模型修正 Model Updating**
*(结构<-->振动)利用频率响应(FR)等数据对有限元模型结构参数进行更新*

<!-- more -->

# 基础知识

## 动力学

[动力学分析之模态分析 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/559497137)

- 固有频率[什么是固有频率？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/23320350) [2.固有频率介绍_固有频率越大越好还是越小越好-CSDN博客](https://blog.csdn.net/qq_39200110/article/details/106057561)
  - 通常一个结构有很多个**固有频率**。固有频率与外界激励没有关系，是结构的一种固有属性。**只与材料的刚度 k 和质量 m 有关**：$\omega_{n} =  \sqrt{\frac{k}{m}}$
- 模态分析是一种处理过程：根据结构的固有特性，包括固有频率、阻尼比和模态振型，这些动力学属性去描述结构。[结构动力学中的模态分析(1) —— 线性系统和频响函数 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/69229246)
  - 每一个模态都对应一个**固定的振动频率，阻尼比**及**振动形式**。而**固定的振动形式**也被称为**模态振型**[模态是什么](https://www.zhihu.com/question/24578439/answer/837484786)
  - 一个自由度为 N 的系统，包含 N 阶模态，通常将固有频率由小到大排列
  - 模态分析方法是通过坐标变换将多自由度系统解耦成为模态空间中的一系列单自由度系统，通过坐标变换可将模态空间中的坐标变换为实际物理坐标得到多自由度系统各个坐标的时域位移解。
  - 我们认为系统某点的响应及频响函数都是全部模态的叠加，即我们所采用的是完整的模态集。但实际上并非所有模态对响应的贡献都是相同的。对低频响应来说，高阶模态的影响较小。对实际结构而言，**我们感兴趣的往往是它的前几阶或十几阶模态**，更高阶的模态常常被抛弃。[机械振动理论(4)-实验实模态分析 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/82442784)
  - 模态参数识别:是指对振动系统进行激振（即输入），**通过测量获得系统的输入、输出**（或仅仅是输出）信号数据，经过对他们进行处理和分析，依据不同的识别模型和方法，**识别出系统的结构模态参数**（如频率、阻尼比、振型、模态刚度、模态质量等）。这类问题为结构动力学第一类逆问题[模态参数识别及有限元模型修正(seu.edu.cn)](https://seugs.seu.edu.cn/_upload/article/files/e0/39/83ad9ffd4cc098c6a6a3f439177e/7cd241da-840e-437a-9bbd-3a0794d1584c.pdf)
    - 模态参数识别是结构动力学中的反问题，它建立在实验的基础上，基于理论与实验相结合的原则，辨识出系统的模态参数，最终实现对系统的改进。通过获取结构的动力学特性，对结构性能进行评价，从而判断结构的可靠性及安全性是否符合要求。


## 不确定性

不确定性根据系统内的不确定性来源可以分为Aleatoric Uncertainty和Epistemic Uncertainty，Aleatoric Uncertainty通常指的是数据不确定性，往往来自于数据本身的randomness或variability，是数据固有的一种属性。Epistemic uncertainty通常指的是模型不确定性或认知不确定性，其不确定性通常来源于缺乏足量信息的支撑。这个挑战中。

> https://zhuanlan.zhihu.com/p/656915794

Uncertainty sources：Parameter uncertainty、Model form uncertainty、Experiment uncertainty
根据参数是否存在认知epistemic和/或选择性(偶然)aleatory不确定性，**将不确定性参数分为四类**：
- 既不具有认知不确定性，也不具有选择性不确定性的参数被表示为具有完全确定值的常数。
- 只有认知不确定性的参数被表示为一个未知但固定的常数，落在预定义的区间内。
- 将仅具有aleatory偶然不确定性的参数表示为具有完全确定的分布性质(如分布格式、均值、方差等)的随机变量。这种完全确定的分布称为“精确概率”。
- 同时具有认知不确定性和选择性不确定性的参数被表示为一个分布性质不完全确定的随机变量，即“不精确概率”。这种不精确的概率由所谓的概率盒(P-box)来建模，其中无限数量的累积分布函数(CDF)曲线构成概率空间中的特定区域。

不确定性模型：不确定性参数的分布/区间/P-box
- BMM(Beta Mixture Model)

[【实验笔记】深度学习中的两种不确定性（上） - 知乎](https://zhuanlan.zhihu.com/p/56986840)

## CNN

- [7大类深度CNN架构创新综述 | 机器之心 (jiqizhixin.com)](https://www.jiqizhixin.com/articles/2019-01-25-6)
- [你应该知道的几种CNN网络与实现 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/176987177)

Other：
- [深度学习全连接与cnn区别 (volcengine.com)](https://www.volcengine.com/theme/1216339-S-7-1)

## 信号分析

- 频谱分析：[从傅里叶变换，到短时傅里叶变换，再到小波分析（CWT），看这一篇就够了（附MATLAB傻瓜式实现代码） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/589651368)
- 经验模态分解EMD [这篇文章能让你明白经验模态分解（EMD）——基础理论篇 - 知乎](https://zhuanlan.zhihu.com/p/40005057)
- 

# 模型修正MU

## 基础知识

[有限元模型修正方法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/400178176)
- 基于动力有限元模型修正：矩阵型修正方法、**设计参数型修正方法**
  - 矩阵型有限元模型修正法是对有限元模型的刚度矩阵和质量矩阵进行直接修正
  - 设计参数型模型修正是对结构的设计参数，如材料的弹性模量，质量密度，截面积，弯曲、扭转惯量等参数进行修正。
- 基于静力有限元模型修正

按照样本数量多少(实验样本数据有时很难获得)
- 随机模型修正(样本数量多)
- 区间模型修正(样本数量少)：
  - Interval model updating is typically performed when gathering data is expensive, time-consuming, or complex and only a limited amount of data is available to perform non-deterministic model updating.
  - 随机模型修正的方法需要大量的实验数据，区间方法被引入，作为一种有用的替代方法来量化不确定参数。Deng等开发了用于更新不确定参数均值和区间半径的两步法。[Interval identification of structural parameters using interval overlap ratio and Monte Carlo simulation](https://readpaper.com/pdf-annotate/note?pdfId=2201610607974204928&noteId=2201611039416835840)

模型修正术语：
- Model updating、Model verification(计算模型是否准确地表示底层数学方程及其解的过程)、Model validation(从其预期用途的角度确定模型准确表示专用物理实验的程度)、Uncertainty quantification、Uncertainty propagation
- interval response surface model (IRSM)


## 基本组成

**不确定性参数/UQ**(待修正参数)
- 设计参数型模型修正：材料参数(E、$\rho$)、结构参数()
- 矩阵型有限元模型修正：有限元模型的质量和刚度矩阵
**模型输出特征**(有限元计算输出)
- 特征值、模态频率、FRF
**有限元模型/Uncertainty propagation**(由于FE计算花费大，大多使用代理模型)
- Surrogate Model
**UQ指标**(计算仿真和实验输出特征之间的差异)
- 特征is分布：
  - 巴氏距离$BD(p(\boldsymbol{x}),q(\boldsymbol{x}))=-\log\int\sqrt{p(\boldsymbol{x})q(\boldsymbol{x})}d\boldsymbol{x}$ or $BD(p_1,p_2)=-\ln\sum_{x\in X}\sqrt{p_1(x)p_2(x)}$
  - 
- 特征is区间or单个定值：
  - 欧式距离
- 子区间相似度(随机和区间都可以使用)
**优化算法**(修正算法)
- SSA、Particle swarm粒子群 optimizer algorithm
- CNN、RNN、MLP......


# 有限元软件

## Nastran

[如何用matlab被nastran给整的明明白白 PART 1 KNOW YOUR ENEMY——.bdf文件 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/33538970)

[Welcome to pyNastran’s documentation for v1.3! — pyNastran 1.3 1.3 documentation (pynastran-git.readthedocs.io)](https://pynastran-git.readthedocs.io/en/1.3/index.html)

### 卫星算例.bdf

不同结构参数生成结构特征量FR

```
$ Elements and Element Properties for region : Shear_Panels
PSHELL   1       1      .003     1               1

- 36  行 .003 Shear_Panels 厚度 theta5
- 429 行 .002 Central_Cylinder 厚度 theta3
- 666 行 .001 Adapter 厚度 theta2 本来应该是密度2.7
- 723 行 .002 Upper_platform 厚度 theta6
- 864 行 .001 Lower_platform 厚度 theta4
- 1020行 7.   mat_N 弹性模量  theta1  
- 1023行 7.   mat_CC
- 1026行 7.   mat_L
- 1029行 7.   mat_SP
- 1032行 7.   mat_U
- 
```

- **主弹性模量**$\theta_1$ 70Gpa，
- **主密度** $\theta_2$  ，密度2.7x $10^{3} kg/m^{3}$ (英文论文) or 适配器厚度 1mm(本 1)
- **中心筒厚度**$\theta_3$ 2mm
- 底板厚度 $\theta_4$ 1mm
- **剪切板厚度**$\theta_5$ 2mm
- 顶板厚度 $\theta_6$ 2.5mm