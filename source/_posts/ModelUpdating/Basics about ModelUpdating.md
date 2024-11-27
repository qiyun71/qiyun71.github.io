---
title: Model Updating Review
date: 2023-12-22 15:03:03
tags: 
categories: ModelUpdating
---

**模型修正 Model Updating**
*(结构<-->振动)通过有限元模型输出响应特征(频率响应等)对有限元模型结构参数进行更新*

有一个异想天开的想法：模型修正直接修正模型的形状 i.e. 模型每个点的位置
- 有限元模型到底是什么？
- 有限元模型在有限元分析中的作用？有限元分析是一个使用最小势能原理来不断尝试得到位移
- 有限元模型的前向计算可不可以进行微分?
  - 可微仿真（Differentiable Simulation）：[\[WIP\] 可微仿真（Differentiable Simulation） - 知乎](https://zhuanlan.zhihu.com/p/566294757?utm_psn=1829181001001201664)
- 

<!-- more -->

# 概念基础

## 动力学

> [动力学分析之模态分析 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/559497137)
> [2.固有频率介绍_固有频率越大越好还是越小越好-CSDN博客](https://blog.csdn.net/qq_39200110/article/details/106057561)
> [固有频率和共振频率的关系是怎样的？ - 知乎](https://www.zhihu.com/question/27833223) 比较基础的解释
> [结构动力学中的模态分析(1) —— 线性系统和频响函数 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/69229246)
> [模态分析软件|模态测试系统|实验模态分析|模态测试软件|模态软件 (hzrad.com)](http://www.hzrad.com/edm-modal-analysis) **模态分析软件**

- 固有频率
  - 通常一个结构有很多个**固有频率**。固有频率与外界激励没有关系，是结构的一种固有属性。**只与材料的刚度 k 和质量 m 有关**：$\omega_{n} =  \sqrt{\frac{k}{m}}$
- 模态分析是一种处理过程：根据结构的固有特性，包括固有频率、阻尼比和模态振型，这些动力学属性去描述结构。
  - 每一个模态都对应一个**固定的振动频率，阻尼比**及**振动形式**。而**固定的振动形式**也被称为**模态振型**[模态是什么](https://www.zhihu.com/question/24578439/answer/837484786)
  - 一个自由度为 N 的系统，包含 N 阶模态，通常将固有频率由小到大排列
  - 模态分析方法是通过坐标变换将多自由度系统解耦成为模态空间中的一系列单自由度系统，通过坐标变换可将模态空间中的坐标变换为实际物理坐标得到多自由度系统各个坐标的时域位移解。
  - 我们认为系统某点的响应及频响函数都是全部模态的叠加，即我们所采用的是完整的模态集。但实际上并非所有模态对响应的贡献都是相同的。对低频响应来说，高阶模态的影响较小。对实际结构而言，**我们感兴趣的往往是它的前几阶或十几阶模态**，更高阶的模态常常被抛弃。[机械振动理论(4)-实验实模态分析 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/82442784)
  - 模态参数识别:是指对振动系统进行激振（即输入），**通过测量获得系统的输入、输出**（或仅仅是输出）信号数据，经过对他们进行处理和分析，依据不同的识别模型和方法，**识别出系统的结构模态参数**（如频率、阻尼比、振型、模态刚度、模态质量等）。这类问题为结构动力学第一类逆问题[模态参数识别及有限元模型修正(seu.edu.cn)](https://seugs.seu.edu.cn/_upload/article/files/e0/39/83ad9ffd4cc098c6a6a3f439177e/7cd241da-840e-437a-9bbd-3a0794d1584c.pdf)
    - 模态参数识别是结构动力学中的反问题，它建立在实验的基础上，基于理论与实验相结合的原则，辨识出系统的模态参数，最终实现对系统的改进。通过获取结构的动力学特性，对结构性能进行评价，从而判断结构的可靠性及安全性是否符合要求。

### 阻尼

> [结构动力学中的阻尼(2) —— 几种常见形式 - 知乎](https://zhuanlan.zhihu.com/p/362801022)

- 粘性阻尼 $F=c \dot{x}$
- 单元阻尼(Element Damping)
- 模态阻尼比(Modal Damping/Modal Damping Ratio)
- 常值阻尼比(Constant Damping Ratio)
- 材料常值阻尼比(Constant Material Damping Coefficient/Ratio)
- 材料结构阻尼系数(Material Structure Damping Coefficient)
- 瑞利阻尼(Rayleigh Damping, **Alpha-Beta阻尼**)，是M和K的线性组合



### 频率信号分析

- 频谱分析：[从傅里叶变换，到短时傅里叶变换，再到小波分析（CWT），看这一篇就够了（附MATLAB傻瓜式实现代码） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/589651368)
- 经验模态分解EMD [这篇文章能让你明白经验模态分解（EMD）——基础理论篇 - 知乎](https://zhuanlan.zhihu.com/p/40005057)

## 不确定性

> [浅谈Epistemic Uncertainty 和 Aleatoric Uncertainty - 知乎](https://zhuanlan.zhihu.com/p/656915794)
> [【实验笔记】深度学习中的两种不确定性（上） - 知乎](https://zhuanlan.zhihu.com/p/56986840)

不确定性根据系统内的不确定性来源可以分为Aleatoric Uncertainty和Epistemic Uncertainty，Aleatoric Uncertainty通常指的是数据不确定性，往往来自于数据本身的randomness或variability，是数据固有的一种属性。Epistemic uncertainty通常指的是模型不确定性或认知不确定性，其不确定性通常来源于缺乏足量信息的支撑。

Uncertainty sources：Parameter uncertainty、Model form uncertainty、Experiment uncertainty
根据参数是否存在认知epistemic和/或选择性(偶然)aleatory不确定性，**将不确定性参数分为四类**：
- 既不具有认知不确定性，也不具有选择性不确定性的参数被表示为具有完全确定值的常数。
- 只有认知不确定性的参数被表示为一个未知但固定的常数，落在预定义的区间内。
- 将仅具有aleatory偶然不确定性的参数表示为具有完全确定的分布性质(如分布格式、均值、方差等)的随机变量。这种完全确定的分布称为“精确概率”。
- 同时具有认知不确定性和选择性不确定性的参数被表示为一个分布性质不完全确定的随机变量，即“不精确概率”。这种不精确的概率由所谓的概率盒(P-box)来建模，其中无限数量的累积分布函数(CDF)曲线构成概率空间中的特定区域。

不确定性模型：不确定性参数的分布/区间/P-box
- BMM(Beta Mixture Model)



# 模型修正MU

## 基础知识

[有限元模型修正方法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/400178176)
- 基于动力有限元模型修正：矩阵型修正方法、**设计参数型修正方法**
  - 矩阵型有限元模型修正法是对有限元模型的刚度矩阵和质量矩阵进行直接修正
  - 设计参数型模型修正是对结构的设计参数，如材料的弹性模量，质量密度，截面积，弯曲、扭转惯量等参数进行修正。
- 基于静力有限元模型修正

模型修正术语：
- Model updating
- Model verification(计算模型是否准确地表示底层数学方程及其解的过程)
- Model validation(从其预期用途的角度确定模型准确表示专用物理实验的程度)
- Uncertainty quantification
- Uncertainty propagation

### Uncertainty in Model updating

> [A feature map of frequency response functions based model updating method using the Bayesian convolutional neural network - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327023007264?ref=pdf_download&fr=RR-2&rr=87df480b9ca904d1)

- theoretical hypothesis
- boundary condition
- geometric properties
- material constants


### 模型修正MU

#### DeterministicMU

> [Deterministic and probabilistic-based model updating of aging steel bridges - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352012423006239#b0005)

Deterministic methodology refers to classical optimization methods, the most extensively adopted in the state of practice. Thus, a vast number of studies can be found in the existing literature using: 
- i) global optimizers, such as genetic algorithms [24], [29], particle swarm [25], evolutionary strategies [17], harmony search [30], or pattern search [31], 
- ii) local optimization algorithms such as the trust-region reflective algorithm [20], [21], 
- iii) hybrid local–global optimization algorithms, such as genetic algorithm and improved cuckoo search (HGAICS) [32], or unscented Kalman filter and harmony search (UKF-HS) [33], 
- iv) surrogate-assisted strategies [34], to mention a few. In this regard, the reader is referred to [10] for a thorough review of the different FE model updating methods and related works.


#### StochasticMU (样本数量多，知道参数的大致分布)

Alternatively, probabilistic methodologies based on Bayesian inference procedures can be employed [35], which have gained increased attention and have experienced significant development in recent years. Thus, in the literature, several research studies can be found using different techniques, such as 
- Markov Chain Monte Carlo (MCMC) with Metropolis-Hastings algorithm [36], [37], 
- Hamiltonian Monte Carlo algorithms [38], [39], 
Transitional MCMC [40], [41], 
Gibbs based approach [42], 
Multiresolution Bayesian non-parametric general regression method (MR-BNGR) [43], 
Approximate Bayesian Computation (ABC) methods [44], [45], 
Variational Bayesian methods [46], [47]


#### IntervalMU (样本数量少，无法得到具体的分布，用区间表示)



## Traditional VS NN-based Method

传统方法(代理模型)：
- 计算量大
- 容易陷入局部最优
- 拟合不足的模型不可避免地引入误差(节省计算成本meta-model)、
- 机械结构维修/替换，环境改变时，模型关键参数改变，传统方法每次都要重新寻优进行修正

基于NN(逆代理模型)：
- NN可以拟合复杂地函数(their powerful generalization ability)
- NN方法在模型关键参数改变后，依然可以快速修正(**我觉得这个的前提是，改变后的模型实验测量地响应依旧可以被训练集所包围**)

NN的训练十分依赖数据集，数据集范围要足够大，将实验的数据包含进来，数据集是根据有限元仿真生成的，如果有限元仿真响应与实验测量响应之间的差异很大的话，训练出来的NN很难对实验测量数据进行很好地修正

~~如果无论如何改变**影响有限元输出的因素(结构参数、网格划分、有限元简化...)**，都无法使得有限元模型可以仿真出与实验相近的结果，则很难进行修正。(*一般工程上建立的有限元模型都很准确吧？与实验测量的数据近似吧？*)~~ : 有限元模型修正应该是建立在有限元模型相对来说比较准确的基础上的

## Traditional Model Updating with Optimization algorithms(OptimAlgo)

Traditional Model Updating is a class of inverse problem, it continually find the optimal structural parameters by OptimAlgo to minimize loss or metrics(don't need differentiable to back propagation).

**不确定性参数/UQ**(待修正参数的不确定性描述(区间 or 随机 or P-box))
- 设计参数型模型修正：材料参数(E、$\rho$)、结构参数()
- 矩阵型有限元模型修正：有限元模型的质量和刚度矩阵
**模型输出特征**(有限元计算输出)
- 特征值、模态频率、FRF高维
**Uncertainty propagation with FE/Surrogate Model**(由于FE计算花费大，大多使用代理模型)
- Uncertainty propagation方法类别：概率理论、模糊数学、区间分析
- CNN、RNN、MLP......
**UQ指标**(计算仿真和实验输出特征之间的差异)
- 特征is分布：
  - 巴氏距离$BD(p(\boldsymbol{x}),q(\boldsymbol{x}))=-\log\int\sqrt{p(\boldsymbol{x})q(\boldsymbol{x})}d\boldsymbol{x}$ or $BD(p_1,p_2)=-\ln\sum_{x\in X}\sqrt{p_1(x)p_2(x)}$
  - 
- 特征is区间or单个定值：
  - 欧式距离
- 子区间相似度(随机和区间都可以使用)
**优化算法**(修正算法)
- SSA、Particle swarm optimizer algorithm(粒子群)

### FE Surrogate Model

- **data-fit** type：conventional response surface method, neural networks and Kriging models
  - 适用于参数是高维的情况
- an efficient **physical-based** low fidelity model：perturbation method
  - 适用于参数是低维的情况

- [ ] Response Surface Model(RSM)
- [ ] Back Propagation(BP) neural networks
- [ ] Radial Basis Function(RBF) neural networks
- [x] Multi Layer Perceptron(MLP) neural networks

### Optimization Algorithms

寻优算法

- [x] Sparrow Search Algorithm(SSA)
- [ ] Particle Swarm Optimization(PSO)
- [ ] Simulated Annealing(SA)
- [ ] Genetic Algorithm(GA)

### UQ Metrics

Loss Functions and Performance Metrics
Loss represents discrepancies between the model simulated outputs and experimental data.

- [X] L1 L2 or MAE MSE or Euclidian distance(ED)
- [x] Bhattacharyya distance(BD)
- [x] Mahalanobis distance(MD)
- [ ] IOR
- [ ] IDD
- [x] Interval Similarity(IS)
- [x] Sub-Interval Similarity(SIS)
- [x] Ellipse Similarity(ES)
- [ ] Sub-Ellipse Similarity(SES)
- [ ] Interval probability box(I-pbox)


## Question

Q1：不确定性传播与FE代理模型的关系？
A1：不确定性传播是指通过一组输入参数，经过FE/代理模型计算，得到一组输出响应。
> Uncertainty propagation is the process of transferring the uncertainty characteristics from the input parameters to the output quantify of interest through the numerical model (or a specific pathway among multiple sub-models thereof). By [Stochastic Model Updating with Uncertainty Quantification_An Overview and Tutorial](Stochastic%20Model%20Updating%20with%20Uncertainty%20Quantification_An%20Overview%20and%20Tutorial.md)


# FEA

[Learn-FEA](Learn-FEA.md)


