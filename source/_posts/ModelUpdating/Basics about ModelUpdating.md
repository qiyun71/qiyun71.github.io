---
title: Model Updating Review
date: 2023-12-22 15:03:03
tags: 
categories: ModelUpdating
---

**模型修正 Model Updating**
*(结构<-->振动)通过有限元模型输出响应特征(频率响应等)对有限元模型结构参数进行更新*

有一个想法：模型修正直接修正模型的形状 i.e. 模型每个点的位置
- 有限元模型到底是什么？
- 有限元模型在有限元分析中的作用？有限元分析是一个使用最小势能原理来不断尝试得到位移
- 有限元模型的前向计算可不可以进行微分?
  - 可微仿真（Differentiable Simulation）：[\[WIP\] 可微仿真（Differentiable Simulation） - 知乎](https://zhuanlan.zhihu.com/p/566294757?utm_psn=1829181001001201664)

<!-- more -->

# 概念基础

广义的“模型”指任一具有输入输出的函数，确认关注的某个量（输出）后，发现影响该量的各种因素（输入），并建立输入与输出之间的映射关系（函数），该映射关系即为“模型”。
https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221pOmQ5YiOVQXs-jaPlDlKOilj4TkEaf5K%22%5D,%22action%22:%22open%22,%22userId%22:%22107403435292964343607%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing

1.  **微观/场级 (Micro/Field Level)**：
    *   **代表**：有限元 (FEM), CFD。
    *   **求解**：PDEs。
    *   **修正目标**：材料本构、边界条件。
2.  **组件/系统级 (Component/System Level)**：
    *   **代表**：多体动力学 (MBD), 1D Amesim/Simulink。
    *   **求解**：ODEs (常微分方程) 或 DAEs (微分代数方程)。
    *   **修正目标**：连接刚度、传递效率、控制参数。
3.  **唯象/经验级 (Phenomenological/Empirical)**：
    *   **代表**：疲劳寿命公式, 电池ECM, 经验回归公式。
    *   **求解**：代数方程。
    *   **修正目标**：公式中的经验系数（Coefficients）、退化因子。
  
| 物理领域    | **函数 $f$ (黑箱)**           | **输入 $X$ (可修正参数)** | **求解目标 $Y_{raw}$ (中间量)** | **输出 $Y_{obs}$ (观测/修正目标)** |
| :------ | :------------------------ | :----------------- | :----------------------- | :------------------------- |
| **动力学** | $M\ddot{x}+C\dot{x}+Kx=F$ | 密度, 弹性模量, 连接刚度     | 节点位移 $u$                 | **固有频率, FRF, 振动加速度**       |
| **热学**  | Fourier定律                 | 热导率, 接触热阻, 对流系数    | 节点温度 $T$                 | **关键点温度, 达到稳态时间**          |
| **流体**  | N-S 方程                    | 粘度, 边界层粗糙度, 入口速度   | 速度 $v$, 压力 $p$           | **阻力/升力, 压降, 流量**          |
| **电磁**  | Maxwell 方程                | B-H曲线, 电导率         | 磁矢势 $A$                  | **电感, 转矩, 效率**             |
| **声学**  | Helmholtz 方程              | 吸声系数, 声速           | 声压 $p$                   | **分贝(dB), 传递损失**           |

有限元（FEM）、CFD等，属于“场求解器”（Field Solvers），它们求解的是物理量在空间和时间上的分布。

除了“场”模型，工程界还有大量“唯象模型”（Phenomenological Models）或“系统级模型”（System-level Models）。这些模型通常不关心物体内部每一点的应力或温度，而是关心宏观性能、寿命、化学反应或系统行为。

| 物理领域                  | **函数 $f$ (黑箱)**                                                  | **输入 $X$ (可修正参数)**                     | **求解目标 $Y_{raw}$ (中间状态)** | **输出 $Y_{obs}$ (观测/修正目标)**              |
| :-------------------- | :--------------------------------------------------------------- | :------------------------------------- | :------------------------ | :-------------------------------------- |
| **车辆动力学**<br>(纵向)     | **功率平衡方程**<br>$F_{trac} = \sum F_{res} + ma$                     | 风阻系数 $C_d$, 滚阻系数 $f$, 传动效率 $\eta$      | 车速 $v(t)$, 加速度 $a(t)$     | **百公里加速时间, 燃油消耗量(L/100km), 最高车速**       |
| **液压系统**<br>(Amesim等) | **流量连续性 + 伯努利**<br>$\dot{P} = \frac{\beta}{V}(Q_{in}-Q_{out})$   | 流量系数 $C_q$, 油液体积模量 $\beta$, 泄漏系数       | 管路压力 $P(t)$, 流量 $Q(t)$    | **油缸伸出速度, 压力超调量, 建立稳态的时间**              |
| **电力电子**<br>(电路仿真)    | **基尔霍夫定律 (KCL/KVL)**<br>状态空间方程                                   | 电感 $L$, 电容 $C$, 开关管导通电阻 $R_{on}$       | 节点电压 $V$, 支路电流 $I$        | **输出电压纹波, 电流谐波(THD), 转换效率**             |
| **热网络**<br>(Lumped)   | **热平衡方程**<br>$C \dot{T} = Q_{in} - \frac{T-T_{amb}}{R_{th}}$     | 热容 $C$, 热阻 $R_{th}$ (对流/接触)            | 节点温度 $T(t)$               | **外壳表面温度, 芯片结温(推算), 达到热平衡时间**           |
| **电池模型**<br>(ECM)     | **Thevenin 等效电路**<br>$\dot{U}_p = -\frac{U_p}{RC} + \frac{I}{C}$ | 欧姆内阻 $R_0$, 极化阻抗 $R_p, C_p$, SOC-OCV曲线 | 极化电压 $U_p$, SOC           | **端电压 $V_{terminal}$ (修正基准), 放电容量(Ah)** |
| **控制系统**<br>(PID)     | **传递函数 / 差分方程**<br>$u = K_p e + K_i \int e$                      | 比例/积分/微分增益 ($K_p, K_i, K_d$)           | 控制量 $u(t)$ (占空比/电流)       | **阶跃响应超调量, 稳态误差, 调节时间**                 |

| 物理领域               | **函数 $f$ (黑箱)**                                                           | **输入 $X$ (可修正参数)**                | **求解目标 $Y_{raw}$ (计算中间量)** | **输出 $Y_{obs}$ (观测/修正目标)**         |
| :----------------- | :------------------------------------------------------------------------ | :-------------------------------- | :------------------------- | :--------------------------------- |
| **疲劳寿命**<br>(S-N法) | **Miner 线性累积损伤**<br>$D = \sum \frac{n_i}{N_i}$                            | 材料常数 (强度系数, 指数 $b$), 表面加工系数 $K_s$ | 累积损伤度 $D$ (无量纲, <1)        | **疲劳寿命 (循环次数 $N_f$), 失效位置**        |
| **断裂力学**<br>(裂纹扩展) | **Paris 公式**<br>$\frac{da}{dN} = C(\Delta K)^m$                           | 裂纹扩展系数 $C, m$, 初始裂纹尺寸 $a_0$       | 裂纹扩展速率 $da/dN$             | **当前裂纹长度 $a(t)$, 断裂前的剩余时间**        |
| **电子老化**<br>(可靠性)  | **Arrhenius 模型**<br>$AF = e^{\frac{E_a}{k}(\frac{1}{T_0} - \frac{1}{T})}$ | 活化能 $E_a$, 加速因子系数                 | 加速因子 $AF$ (倍率)             | **平均故障间隔时间 (MTBF), 失效率 $\lambda$** |
| **光学设计**<br>(光线追踪) | **几何光学 (反射/折射)**<br>Monte Carlo 模拟                                        | 表面粗糙度 (BSDF参数), 材料折射率 $n$         | 光线路径, 光子击中数                | **照度值 (Lux), 光斑均匀度, 黄斑/蓝边现象**      |
| **交通流**<br>(宏观)    | **Greenshields 模型**<br>$v = v_f (1 - k/k_j)$                              | 自由流速度 $v_f$, 阻塞密度 $k_j$           | 交通流密度 $k$                  | **平均车速, 道路通行能力 (辆/小时)**            |
| **经济/成本**<br>(参数化) | **学习曲线 (Learning Curve)**<br>$C_n = C_1 n^{-b}$                           | 学习率系数 $b$, 首件成本 $C_1$             | 第 $n$ 件的单形成本               | **量产后的平均成本, 盈亏平衡点**                |

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

[Uncertainty](../Other%20Interest/Uncertainty.md)


不确定性模型：不确定性参数的分布/区间/P-box
- BMM(Beta Mixture Model)


# 模型修正MU


| 步骤                 | 举例——结构动力学                                                                         |
| ------------------ | --------------------------------------------------------------------------------- |
| 目的                 | 为了避免设计的结构发生共振，需确保仿真模型的频率预测准确。                                                     |
| 确定数值模型             | 结构动力学有限元分析                                                                        |
| 选择输出特征<br>(根据实验测量) | 前n 阶模态频率<br>(实验方法：锤击法或激振器法，通过 FFT 识别频响函数峰值)                                       |
| 选择输入参数<br>(敏感性分析)  | 对频率最敏感的结构参数<br>(例如：杨氏模量E、厚度 t、材料密度ρ)                                              |
| 模型修正               | 构建目标函数并优化                                                                         |
| 模型验证               | 模态置信度 (MAC) 检验不仅频率误差要小于阈值（如 <3%），还需计算仿真振型与实验振型的 MAC 值（应 >0.9），确保“频率对上了，振动形状也是对的”。 |


## 基础知识

> [有限元模型修正方法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/400178176)

分类：
- 基于动力有限元模型修正：矩阵型修正方法、**设计参数型修正方法**
  - 矩阵型有限元模型修正法是对有限元模型的刚度矩阵和质量矩阵进行直接修正
  - 设计参数型模型修正是对结构的设计参数，如材料的弹性模量，质量密度，截面积，弯曲、扭转惯量等参数进行修正。
- 基于静力有限元模型修正

***目前主流/常说的模型修正就是设计参数型模型修正***


模型修正术语：
- **Model updating**“adjusting physical or non-physical parameters in the computational model to improve agreement with experimental results.” ([Bi 等, 2023, p. 2](zotero://select/library/items/5JEKED2M)) ([pdf](zotero://open-pdf/library/items/5Y239HYU?page=2&annotation=C2N2YWH8)) 🔤调整计算模型中的物理或非物理参数，以提高与实验结果的一致性。🔤
- **Model verification**“a computational model accurately represents the underlying mathematical equation and its solution.” ([Bi 等, 2023, p. 2](zotero://select/library/items/5JEKED2M)) ([pdf](zotero://open-pdf/library/items/5Y239HYU?page=2&annotation=WRH5KXHU)) 🔤计算模型准确地表示基础数学方程及其解。🔤
- **Model validation**“the degree to which the model is an accurate representation of dedicated physical experiments from the perspective of its intended use” ([Bi 等, 2023, p. 2](zotero://select/library/items/5JEKED2M)) ([pdf](zotero://open-pdf/library/items/5Y239HYU?page=2&annotation=NHPQNNUH)) 🔤从预期用途的角度来看，模型准确表示专用物理实验的程度🔤
- **Uncertainty quantification**“characterising all uncertainties in the model or experiment and of quantifying their effect on the simulation or experimental outputs” ([Bi 等, 2023, p. 2](zotero://select/library/items/5JEKED2M)) ([pdf](zotero://open-pdf/library/items/5Y239HYU?page=2&annotation=LUSTGYTW)) 🔤描述模型或实验中的所有不确定性，并量化它们对模拟或实验输出的影响🔤
- **Uncertainty propagation**“transferring the uncertainty characteristics from the input parameters to the output quantify of interest through the numerical model (or a specific pathway among multiple sub-models thereof).” ([Bi 等, 2023, p. 2](zotero://select/library/items/5JEKED2M)) ([pdf](zotero://open-pdf/library/items/5Y239HYU?page=2&annotation=5LPY7VRR)) 🔤通过数值模型（或其多个子模型之间的特定路径）将不确定性特性从输入参数传输到感兴趣的输出量化。🔤

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
> Uncertainty propagation is the process of transferring the uncertainty characteristics from the input parameters to the output quantify of interest through the numerical model (or a specific pathway among multiple sub-models thereof). By [Stochastic Model Updating with Uncertainty Quantification_An Overview and Tutorial](Review/Stochastic%20Model%20Updating%20with%20Uncertainty%20Quantification_An%20Overview%20and%20Tutorial.md)


# FEA

[Learn-FEA](Learn-FEA.md)


