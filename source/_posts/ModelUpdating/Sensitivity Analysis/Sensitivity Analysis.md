---
title: Sensitivity Analysis
date: 2025-06-17 13:17:17
tags:
  - 
categories: ModelUpdating/Method
---

Sensitivity Analysis
- LSA
- GSA

<!-- more -->

# Theory

## LSA

Local sensitivity analysis：(The failure-probability-based local sensitivity)
- the moment-based methods
- the sampling-based methods
- the surrogate-based methods

## GSA

全局敏感性分析（GSA）是一种通过使用计算机辅助完成整体计算分析的强大技术，用于分析输入不确定性对模型输出结果变化的影响。

“Global Sensitivity Analysis methods” ([Sadeghi和Matwin, 2024, p. 2](zotero://select/library/items/9J5T3BYP)) ([pdf](zotero://open-pdf/library/items/CHJHEB36?page=2&annotation=GIMKIAKV))
GSA 分类：
- Variance based methods
  - Sobol *常用方法*
  - The Fourier amplitude sensitivity test(FAST)
  - random balance design(RBD) and FAST RBD
-  Derivative based methods
  - Morris
  - DGSM
- Distribution based methods(moment independent-based methods)
  - PAWN
  - DELTA
- Others：quantile-based  methods、 elementary effect method、failure probability-based methods

### Sobol方法

输入$\mathbf{x}=\left({x}_{1},{x}_{2},\cdots ,{x}_{n}\right)$，系统输出$y=f(x)$，在假设输入参数相互独立的前提下，输出响应$y$的总方差为：

$$\mathrm{V}\mathrm{a}\mathrm{r}\left(y\right)=\sum _{i=1}^{n}{\mathrm{V}\mathrm{a}\mathrm{r}}_{i}\left(y\right)+\sum _{1\le i<j\le n}{\mathrm{V}\mathrm{a}\mathrm{r}}_{ij}\left(y\right)+\cdots +{\mathrm{V}\mathrm{a}\mathrm{r}}_{12\cdots n}\left(y\right)$$
- ${\mathrm{V}\mathrm{a}\mathrm{r}}_{i}\left(y\right)$ 表示仅由参数$x_{i}$引起的输出方差
- ${\mathrm{V}\mathrm{a}\mathrm{r}}_{ij}\left(y\right)$ 表示$x_{i}$与$x_j$交互作用引起的输出方差
- ${\mathrm{V}\mathrm{a}\mathrm{r}}_{12\cdots n}\left(y\right)$ 表示所有参数联合作用产生的高阶交互影响

一阶Sobol指标 ${S}_{i}=\frac{{E}_{-i}\left[{\mathrm{V}\mathrm{a}\mathrm{r}}_{i}\left(y\right)\right]}{\mathrm{V}\mathrm{a}\mathrm{r}\left(y\right)}$， 用于衡量单个参数$x_{i}$对输出方差的贡献

总阶Sobol指标  ${S}_{{T}_{i}}=1-\frac{{E}_{i}\left[Va{r}_{-i}\left(y\right)\right]}{Var\left(y\right)}$ 用于衡量包含参数$x_{i}$的所有阶次（包括其自身及与其他参数的交互作用）对输出方差的贡献

---WIKI source

> [Variance-based sensitivity analysis - Wikipedia](https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis)
> [Sobol敏感性分析详解-CSDN博客](https://blog.csdn.net/xiaosebi1111/article/details/46517409)

从黑箱的角度来看，任何模型可以被表示为一个函数：$Y=f(X)$，输入X为一个d维向量，Y为所感兴趣的标量(例如某个模态频率值)。其中假设输入在单位“超立方体”内独立均匀分布，即$X_i\in[0,1]$

函数$f(X)$可以被分解为：$Y=f_0+\sum_{i=1}^df_i(X_i)+\sum_{i<j}^df_{ij}(X_i,X_j)+\cdots+f_{1,2,\ldots,d}(X_1,X_2,\ldots,X_d)$
- $f_{0}$为常数
- $f_{i}$为$X_{i}$的函数
- $f_{ij}$为$X_{i},X_{j}$的函数

并且所有的函数满足正交的特性，即：$\int_0^1f_{i_1i_2\ldots i_s}(X_{i_1},X_{i_2},\ldots,X_{i_s})dX_k=0,\mathrm{~for~}k=i_1,\ldots,i_s$，则可以从期望的角度来描述

一阶：$S_i=\frac{V_i}{\mathrm{Var}(Y)} =\frac{\mathrm{Var}_{X_i}(E_{\mathbf{X}_{\sim i}}(Y\mid X_i))}{\mathrm{Var}(Y)}$ 
1. 固定$X_{i}$，变化$X_{\sim i}$，得到一组Y，求Y的期望$E_{X_{\sim i}}(Y|X_{i})$
2. 变化$X_{i}$，$E_{X_{\sim i}}(Y|X_{i})$随之变化，求其方差，即为$X_{i}$单独作用下的效应

总阶：$S_{Ti}=\frac{E_{\mathbf{X}_{\sim i}}\left(\mathrm{Var}_{X_i}\left(Y\mid\mathbf{X}_{\sim i}\right)\right)}{\mathrm{Var}(Y)}=1-\frac{\mathrm{Var}_{\mathbf{X}_{\sim i}}\left(E_{X_i}\left(Y\mid\mathbf{X}_{\sim i}\right)\right)}{\mathrm{Var}(Y)}$
1. 固定$X_{\sim i}$

Procedure：N为样本数量，d为自变量个数
- 生成`Nx2d` 样本矩阵，前d列为$A$，后d列为$B$
- 用矩阵$B$中的第i列替换矩阵$A$的第i列，生成$AB^{i}$ 矩阵(`Nxd`)
- 得到d+2个矩阵（$A,B,AB^{i}(i=1,2,\dots d)$），每个矩阵根据d个自变量获得输出y，总共有$N*(d+2)$个样本点($x_{1},x_{2}\dots x_{n} \rightarrow y$)
- $\mathrm{Var}(Y)=var(cat(A,B))$，对AB两矩阵得到的$N*2$个样本点求方差
- $\mathrm{Var}_{X_i}\left(E_{\mathbf{X}_{\sim i}}\left(Y|X_i\right)\right)\approx\frac{1}{N}\sum_{j=1}^Nf(\mathbf{B})_j\left(f{\left(\mathbf{A}_B^i\right)}_j-f{\left(\mathbf{A}\right)}_j\right)$
- 一阶Sobol指标 $S_{i}=\frac{Var_{X_{i}}(E_{X\sim i}(Y|X_{i}))}{Var(Y)}$
- $E_{\mathbf{X}_{\sim i}}\left(\mathrm{Var}_{X_i}\left(Y\mid\mathbf{X}_{\sim i}\right)\right)\approx\frac{1}{2N}\sum_{j=1}^N\left(f(\mathbf{A})_j-f{\left(\mathbf{A}_B^i\right)_j}\right)^2$
- 总阶Sobol指标 $S_{Ti}=\frac{E_{X\sim i}(Var_{X_{i}}(Y|X_{\sim i}))}{Var(Y)}$
- 不常用的二阶Sobol指标：
  - $V_{ij}=\frac{1}{N}\sum_{q=1}^{N}f(\mathbf{B})_{q}\left(f(\mathbf{A}_{B}^{ij})_{q}-f(\mathbf{A}_{B}^{i})_{q}-f(\mathbf{A}_{B}^{j})_{q}+f(\mathbf{A})_{q}\right)$
  - $S_{ij}=\frac{V_{ij}}{\mathrm{Var}(Y)}$

WIKI source---

# Paper

| Year     | Title                                                                                                                                                                                      | Corresponding                                               | 背景                             | 方法                                                                          | 不足                                                                        |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------ | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **2013** | An application of the Kriging method in global sensitivity analysis with parameter uncertainty                                                                                             | Pan Wang<br>Northwestern Polytechnical University           | 混合不确定性下的**失效概率**灵敏度，模型计算效率低    | Kriging 替代三重循环采样                                                            |                                                                           |
| 2014     | A **review** on global sensitivity analysis methods                                                                                                                                        |                                                             | 全局灵敏度分析（GSA）方法分类与适用性综述         | 回归/平滑/统计学习/Monte Carlo                                                      |                                                                           |
| 2024     | [A dimension reduction-based Kriging modeling method for high-dimensional time-variant uncertainty propagation and global sensitivity analysis](A%20dimension%20reduction-based%20Kriging%20modeling%20method%20for%20high-dimensional%20time-variant%20uncertainty%20propagation%20and%20global%20sensitivity%20analysis.md) | Zhouzhou Song<br>Shanghai Jiao Tong University              | 少样本下为高维输入输出的时变问题构建代理模型         | SVD 输出降维 + ISDR 嵌入Kriging降维输入 + 维度估计器 + 广义sobol指数                           | PCA和ISDR都是线性降维方法，未验证强非线性系统适应性                                             |
| 2024     | A novel single-loop estimation method for predictive **failure probability**-based global sensitivity analysis                                                                             | Zhao Zhao<br>National University of Singapore               | 预测失效概率（PFP）灵敏度计算效率低            | AK-SL-MCS<br>singleloop estimation method<br>                               |                                                                           |
| 2022     | A unified approach for global sensitivity analysis based on **active subspace and Kriging**                                                                                                | Changcong Zhou<br>Northwestern Polytechnical University     | 统一框架实现多维灵敏度分析                  | 主动子空间 + Kriging 代理 + DGSM/Sobol 指数                                          |                                                                           |
| 2024     | Active Learning of **Ensemble Polynomial Chaos Expansion Method** for Global Sensitivity Analysis                                                                                          | Xiaobing Shang<br>Harbin Engineering University             | 模型 Sobol 指数计算成本高               | EPCE ensemble polynomial chaos expansion (EPCE) surrogate model  + 混合主动学习准则 |                                                                           |
| 2025     | Adaptive Kriging *elastodynamic* modelling and analysis for a redundantly actuated parallel manipulator                                                                                    | Tengfei Tang<br>Zhejiang Sci-Tech University                | 冗余驱动并联机构**弹性动力学**性能量化          | 自适应 Kriging + 频率变化指标                                                        |                                                                           |
| 2021     | Advanced kriging-based surrogate modelling and sensitivity analysis for *rotordynamics* with uncertainties                                                                                 | E. Denimal                                                  | **转子动力学**多源不确定性临界转速预测          | 高级 Kriging 代理模型                                                             |                                                                           |
| 2020     | An active learning Kriging model combined with directional importance sampling method for efficient **reliability analysis**                                                               | Qing Guo<br>Northwestern Polytechnical University           | 小失效概率与多失效模式系统可靠性，计算效率低         | 方向重要性采样（DIS）+ Kriging 主动学习                                                  |                                                                           |
| 2022     | An adaptive sampling method for Kriging surrogate model with **multiple outputs**                                                                                                          | Zhangming Zhai<br>National University of Defense Technology | **多输出 Kriging 模型样本空间优化，学习效率低** | Delaunay 三角剖分 + TOPSIS 权重分配                                                 |                                                                           |
| 2021     | An efficient and robust Kriging-based method for system **reliability analysis**                                                                                                           | Jian Wang<br>Northeastern University                        | 多失效模式系统可靠性计算                   | SAIS-SYS 策略 + Kriging 误差度量                                                  | 对于高维问题和不可微分甚至不连续的性能函数问题有一些局限性。                                            |
| 2022     | An efficient method by nesting adaptive Kriging into Importance Sampling for **failure-probability-based** global sensitivity analysis                                                     | Jingyu Lei<br>Northwestern Polytechnical University         | FP-GS的估计效率低                    | AK-IS嵌套 + Metropolis–Hastings采样                                             | multi-MPP 任务不适用<br>the most probable failure point (MPP)                  |
| 2023     | An efficient multi-fidelity Kriging surrogate model-based method for global sensitivity analysis                                                                                           | Xiaobing Shang<br>Harbin Engineering University             | 模型评估成本高                        | CoKriging + 方差解析表达式**降维**                                                   | 不考虑梯度信息                                                                   |
| 2024     | An improved adaptive Kriging model for importance sampling reliability and **reliability** global sensitivity analysis                                                                     | Da-Wei Jia<br>Northwestern Polytechnical University         | 主动学习过程的效率低                     | IS密度融入学习函数 + 惩罚函数                                                           |                                                                           |
| 2023     | Bayesian Model Updating for Structural Dynamic Applications Combing Differential Evolution Adaptive Metropolis and Kriging Model                                                           | Jice Zeng<br>Univ. of Louisville,                           | **高维**问题，高计算成本                 | DREAM算法 + Kriging代理 + 全局灵敏度降维                                               | 更新大型斜拉桥模型和保持物理值方面的能力有限                                                    |
| 2019     | Efficient global sensitivity analysis for flow-induced vibration of a *nuclear reactor assembly* using Kriging surrogates                                                                  | Gregory A. Banyay<br>University of Pittsburgh               | 计算成本                           | Kriging代理 + GSA                                                             | 应用于具有非线性（即非线性弹簧和阻尼器）和非平稳载荷（即冷却剂事故声学或地震载荷）                                 |
| 2020     | Global **reliability sensitivity analysis** by Sobol-based dynamic adaptive kriging importance sampling                                                                                    | Francesco Cadini<br>Politecnico di Milano                   | 罕见事件故障概率，计算成本高                 | Sobol动态自适应Kriging重要性采样                                                      | DoE的样本数量增加引入数值问题、维数灾难、<br>停止准则设置导致优化阶段长、要求为全局灵敏度分析所需的每个失效概率估计值的最大变异系数设置阈值 |
| 2024     | Global sensitivity analysis for **multivariate outputs** using generalized RBF-PCE metamodel enhanced by variance-based sequential sampling                                                | Lin Chen<br>Sun Yat-sen University                          | 输出系统协方差基灵敏度分析                  | RBF-PCE混合模型 + 方差序列采样                                                        | 多变量输出的dependent-input模型的全局敏感性分析和高维输出模型的顺序采样技术                             |
| 2022     | Global Sensitivity Analysis Using a Kriging Metamodel for *EM Design Problems* With Functional Outputs                                                                                     | Arnold Bingler                                              | 计算成本高                          | Kriging代理 + 自适应采样                                                           |                                                                           |
| 2016     | Global sensitivity analysis-enhanced surrogate (GSAS) modeling for **reliability** analysis                                                                                                | Zhen Hu<br>Vanderbilt University                            | 新训练点的选择问题                      | 全局灵敏度引导代理训练（GSAS）                                                           | 将所提方法与重要性采样相结合，可以进一步提高所提方法的计算效率                                           |
| 2020     | Kriging based **reliability** and sensitivity analysis – Application to the stability of an earth dam                                                                                      | Xiangfeng Guo<br>Grenoble INP                               | 计算成本高                          | MCS+GSA+FORM                                                                |                                                                           |
| 2022     | Kriging‐based analytical technique for global sensitivity analysis of systems with **multivariate output**                                                                                 | Yushan Liu<br>                                              | 多变量输出的系统                       | K-based analysis 估计多变量敏感性指数MSI                                              | 均匀采样构建Kriging，对于严重震荡输出的系统无法很好处理                                           |
| 2019     | Parameter selection for model updating with global sensitivity analysis                                                                                                                    | Zhaoxu Yuan<br>Harbin Institute of Technology               | 没有任何方法可以保证“正确”的选择模型修正的参数       | 新的评估函数和复合型敏感性指标                                                             |                                                                           |
| 2021     | Random forests for global sensitivity analysis: A selective **review**                                                                                                                     |                                                             |                                | 随机森林+GSA的综述                                                                 |                                                                           |
| 2023     | **Reliability** and global sensitivity analysis based on importance directional sampling and adaptive Kriging model                                                                        | Da‐Wei Jia<br>Northwestern Polytechnical University         |                                | 重要性方向采样（IDS） + 辅助区域停止准则<br>AK-IDS-RGS<br>Reliability and Global Sensitivity |                                                                           |
| 2021     | **Reliability** and sensitivity analysis of composite structures by an adaptive Kriging based approach                                                                                     | Changcong Zhou<br>Northwestern Polytechnical University     | 复合材料结构的可靠性和敏感性分析               | 适应Kriging（U函数） + 局部/全局敏感度                                                   |                                                                           |
| 2024     | **Reliability**-based design optimization scheme of isolation capacity of nonlinear vibration isolators                                                                                    | Huizhen Liu<br>Northeastern University                      |                                | 全局灵敏度筛选参数 + Kriging代理                                                       |                                                                           |
| 2015     | Sensitivity analysis and Kriging based models for robust stability analysis of *brake systems*                                                                                             | L. Nechak                                                   |                                | GSA+Kriging                                                                 |                                                                           |
| 2025     | Structural **reliability** analysis of high-speed motorized spindle under thermal error based on dynamically adjusted adaptive Kriging model                                               | Rundong Shi                                                 |                                | 多物理场耦合模型 + DAA-WEI Kriging                                                  |                                                                           |
| 2020     | Surrogate-assisted global sensitivity analysis: an **overview**                                                                                                                            | Kai Cheng<br>Northwestern Polytechnical University          | 代理模型在GSA中的选择与应用                | PR/HDMR/PCE/Kriging等方法对比                                                    | 缺乏统一模型选择准则                                                                |
| 2023     | Time-variant **reliability** global sensitivity analysis with single-loop Kriging model combined with importance sampling                                                                  | Qing Guo<br>Xi’an Modern Chemistry Research Institute       | 时变系统失效概率的全局灵敏度                 | 单循环Kriging + 重要性采样 + 贝叶斯公式                                                  | 长时间历程计算成本高                                                                |


## Introduction

Terminology：
Global Sensitivity Analysis (GSA)
Sensitivity Indices (SI)
Partial least squares (PLS)
Principal components Analysis (PCA)
Importance sampling (IS)
subset simulation (SS)******
line sampling (LS)
directional sampling (DS)
Adaptive Kriging （AK）
Failure Probability (FP)
radial basis function (RBF)
sparse polynomial chaos expansion (PCE)
derivative-based global sensitivity measure (DGSM)

Parameter selection for model updating with global sensitivity analysis “Introduction” ([Yuan 等, 2019, p. 483](zotero://select/library/items/EQILHZZE)) ([pdf](zotero://open-pdf/library/items/8AK88VPN?page=1&annotation=X6BWM3UN))
FEMU，参数选择问题
- 早期依赖工程经验，*太主观*
  - LSA，*仅适用于接近初始参数估计值的空间，且仅能识别敏感参数，而无法定位模型不确定性(其不考虑测试数据)*
    - 基于统计的参数选择方法，其中最常用的GSA，*sobol法需要MC大量采样，成本高*
      - 缩放输出协方差矩阵的分解,*无法区分准确建模和未准确建模的参数不确定性*


高维问题的代理模型综述：

![afaeea7917189bb56cf70d5f2adea405.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/afaeea7917189bb56cf70d5f2adea405.png)


### 降维方法(Dimension reduction)

[A dimension reduction-based Kriging modeling method for high-dimensional time-variant uncertainty propagation and global sensitivity analysis](../../../../ModelUpdating/Sensitivity%20Analysis/A%20dimension%20reduction-based%20Kriging%20modeling%20method%20for%20high-dimensional%20time-variant%20uncertainty%20propagation%20and%20global%20sensitivity%20analysis.md) “Introduction” ([Song 等, 2024, p. 2](zotero://select/library/items/I2A9MG56)) ([pdf](zotero://open-pdf/library/items/2TRZV9MT?page=2&annotation=P384ETV4)) ⭐⭐⭐
**不确定性分析**（结构/系统的性能受到源自材料、制造、负载条件和环境等各种不确定性的影响）
- 一般的不确定性传播 + GSA技术， *无法适用于时变问题（高维的输入和输出）* 
  - 基于MC的不确定性传播，*效率问题* --> 代理模型，*维数灾难*--> **维度减少技术**
    - **Kriging代理模型**可以在少样本下量化模型不确定性 + **维度减少技术(降维输入)**，*对高维输出的研究较少*
      - Kriging + 输入输出维度减少技术 （PCA、PLS），*1）主成分数量的确定、2）Kriging模型构建直接训练低维潜在空间的输入输出* 
      - 代理模型用于高维输出的GSA面临挑战，*1）量化输入对整体时变响应的影响，如何将其于每个时间节点的SI联系起来，2）如何开发一个integrated 基于代理的框架来有效估计时变SI*

A unified approach for global sensitivity analysis based on active subspace and Kriging “Introduction” ([Zhou 等, 2022, p. 1](zotero://select/library/items/WYUCSUFY)) ([pdf](zotero://open-pdf/library/items/I78RP8RT?page=1&annotation=ARA5AE6C)) *仅考虑了高维输入*
敏感性分析 LSA --> GSA  “including GSA index based on reliability [6], variance-based sensitivity index [7], moment-independent importance measures [8], quantile based global sensitivity measures [9]” ([Zhou 等, 2022, p. 1](zotero://select/library/items/WYUCSUFY)) ([pdf](zotero://open-pdf/library/items/I78RP8RT?page=1&annotation=HTHF6ALQ))
- the variance-based method of Sobol’ sensitivity indices and DGSM，*MCS计算成本高*
  - 代理模型方法，*维度诅咒，高维输入时训练需要更多的计算*
    - active subspace将其他输入看作是主要输入(activate subspace)的线性融合，通过特征值分解将model在子空间变为低维问题
- 方法框架：响应梯度的协方差矩阵--> DGSM，矩阵特征值分解辨识出active subspace --> activity score，在active subspace(active variables and original outputs)训练好AK --> total effect indices

Global sensitivity analysis for multivariate outputs using generalized RBF-PCE metamodel enhanced by variance-based sequential sampling “Introduction” ([Chen和Huang, 2024, p. 381](zotero://select/library/items/FWX8ZPD9)) ([pdf](zotero://open-pdf/library/items/ZALLWHZD?page=1&annotation=3NXN7GDE))
**Multivariate outputs(就是高维问题)**
复杂数值模型，*存在许多不确定性(真实系统本身，输入不确定性，meta-model建模误差)*
- GSA用于量化输入不确定性对输出不确定性的影响，*传统方法仅用于scalar output* “including elementary effect method [8], derivative-based method [9,10], variance-based method [11] and moment-independent approach” ([Chen和Huang, 2024, p. 381](zotero://select/library/items/FWX8ZPD9)) ([pdf](zotero://open-pdf/library/items/ZALLWHZD?page=1&annotation=Z3VMY4AY))
  - GSC+MCS，*计算成本高*
    - 代理模型，*多变量代理模型准确度不足*
      - RBF-PCE-based method
      - 代理模型训练时新样本的添加
      - *真实多输出响应之间相关性导致敏感性分析冗余，且无法适用于描述多变量输出的综合不确定性*
        - 将输出降维，*对于时变问题，在每个时刻执行GSA导致大量的评估次数*

### 先进采样方法(Adaptive kriging)

An improved adaptive Kriging model for importance sampling reliability and reliability global sensitivity analysis “Introduction” ([Jia和Wu, 2024, p. 1](zotero://select/library/items/866Y2G2M)) ([pdf](zotero://open-pdf/library/items/DZM6NFYY?page=1&annotation=7FQDZPRX))
可靠性和可靠性GSA理论
- 可靠性GSA方法：“variance-based methods [3], moment independent-based methods [4] *可靠性分析关注结构的两种状态——是/否可靠，性能函数(黑箱)没有包含符号/公式，这两种方法不适用* and failure probability-based methods [5].” ([Jia和Wu, 2024, p. 1](zotero://select/library/items/866Y2G2M)) ([pdf](zotero://open-pdf/library/items/DZM6NFYY?page=1&annotation=BCWI89BH)) **基于故障概率的方法侧重于固定输入变量时失效概率的平均变化** 该方法是在获得失效概率和失效样本集后进行的。
- MC和IS数值仿真方法，*为了结果更精确，计算成本高*
  - 代理模型，最常用的是 AK model， 如AK-MCS方法，*不适用于小失效概率问题，需要的样本数量太多*
  - AK + IS/SS/LS/DS，最常用的是 AK-IS
  - 学习函数是AK的核心，U函数及其改进方法最常用，在测量所有候选样本的误分类概率基础上建立，可以选择在极限状态边界的样本点，*U函数的停止准则过于保守，且其都是基于AK预测的均值和标准差建立的，没有考虑IS密度函数的分布特征和输入变量的联合概率密度函数* *-->从而导致样本准确获取困难，迭代次数变多*

Reliability and global sensitivity analysis based on **importance directional sampling** and adaptive Kriging model “Introduction” ([Jia和Wu, 2023, p. 1](zotero://select/library/items/PKXKVNP5)) ([pdf](zotero://open-pdf/library/items/3KY8LIBG?page=1&annotation=MK4ZFUG3))
可靠性和GSA理论
- 可靠性方法(数值仿真/矩估计方法)、GSA方法(variance-based、moment independent-based、failure probability-based)，*数值仿真方法效率低*
  - 代理模型，AK-MCS融合U函数学习和MC方法，*基于MC的主动学习需要样本数量多，尤其是小失效概率问题*
    - AK + IS/SS/LS/DS。“Compared with IS, DS can reduce the dimension of variable space” ([Jia和Wu, 2023, p. 2](zotero://select/library/items/PKXKVNP5)) ([pdf](zotero://open-pdf/library/items/3KY8LIBG?page=2&annotation=X47Q29RM))，AK-IS + AK-DS = AK-IDS，*U函数停止准则过于保守*


Global reliability sensitivity analysis by Sobol-based dynamic adaptive kriging importance sampling “Introduction” ([Cadini 等, 2020, p. 1](zotero://select/library/items/GTC9HZGR)) ([pdf](zotero://open-pdf/library/items/F3I7WHJV?page=1&annotation=BSQDX6A8))
可靠性和风险分析 (评估不确定性对系统影响)
- MC的故障率估计方法，*现代结构/系统非常可靠，故障率低，导致需要大量样本，尤其是性能函数依托FEM计算非常复杂时*
  - 优化算法+MC variance reduction techniques（IS、SS、LS...）+有限元代理模型，以 AK-IS 为参考。*AK的局限，特别是处理非常不规则的极限状态，或者不连续且病态表现的性能函数*
- *不管采用那种方法来估计失效概率FP，结果的鲁棒性取决于认知不确定性，而不是基本随机变量描述的随机不确定性* 量化这些不确定性对系统结构可靠性的影响(可靠性敏感性分析)
  - 敏感性分析评估不确定性参数特征对FP的影响，*需要大量模型评估，计算成本高*
    - 利用FP偏导数进行可靠性敏感性分析等方法，*然而仅可以分析局部敏感性——只关注某些输入参数在名义值附近的微小变动(固定其他参数)*
      - GSA，*计算成本*
        - Kriging meta-modeling with efficient sobol method，*然而仍需要大量计算性能函数，且没有减少代理模型近似误差的策略*

An efficient method by nesting adaptive Kriging into Importance Sampling for failure‐probability‐based global sensitivity analysis “Introduction” ([Lei 等, 2022, p. 3595](zotero://select/library/items/TWFAH56G)) ([pdf](zotero://open-pdf/library/items/R65FWMJV?page=1&annotation=DLW556EE))
不确定性分析（FP sensitivity analysis is classified into FP-LSA and FP-GSA）
- FP-LSA： estimated by the momentbased methods | the sampling-based methods (MCS、IS、SS、LS) | the surrogate-based methods (RSM、SVM、NN、AK)
- FP-GSA...，直接估计FP-GS是一个 double-loop MCS，*非常耗时，尤其是小失效概率和性能函数的公式无法表示(黑箱)的情况下*


### 多保真度代理模型(Multi-fidelity)

An efficient multi-fidelity Kriging surrogate model-based method for global sensitivity analysis “Introduction” ([Shang 等, 2023, p. 1](zotero://select/library/items/APWQBCLI)) ([pdf](zotero://open-pdf/library/items/4FTIT5GX?page=1&annotation=DNR56CZQ))
有限元仿真代替物理系统，*输入参数无法正确认知*
- SA来量化每个输入对输出的影响，可分为LSA和GSA，*其中LSA对于非线性模型提供的敏感性信息有限*
  - GSA中最常用的是sobol index法，传统的sobol适用MCS，*计算成本高* “GSA method consists of three groups: variance decomposition method [10], moment independent method [11], and derivative-based global sensitivity measure method” ([Shang 等, 2023, p. 1](zotero://select/library/items/APWQBCLI)) ([pdf](zotero://open-pdf/library/items/4FTIT5GX?page=1&annotation=Z4FD5ZA9))
    - 另一种是 Fourier Amplitude Sensitivity Test，可以降维，从而降低model复杂度，提高计算效率
    - 或者使用有限元代理模型，传统代理模型通过单保真度计算模型构建，*无法保证足够的精度(FE评估次数有限 or QoI通过不同保真度模型获得)*
      - Multi-fidelity surrogate model，*PCE方法的正交多项式由输入变量的分布决定*

## Case

### Simple case

“A mathematical problem” ([Song 等, 2024, p. 13](zotero://select/library/items/I2A9MG56)) ([pdf](zotero://open-pdf/library/items/2TRZV9MT?page=13&annotation=99YR5DQ4)) ***输入40输出101***

This example considers a nonlinear analytical function with **40 inputs** as follows：
$\mathbf{Y}(t)=\sin(4\pi t)\sum_{i=1}^{25}\left(\frac{X_i+a_i}{1+a_i}\right)+\frac{1}{25}\sum_{i=26}^{30}\left(X_i-5\pi X_it\right)+\cos(5\pi t)\mathrm{atan}\left(t+\sum_{i=31}^{40}X_i^3b_{i-30}\right),$

“a series system with two components” ([Zhao 等, 2024, p. 7](zotero://select/library/items/G7ITCE29)) ([pdf](zotero://open-pdf/library/items/9PPVC2N3?page=7&annotation=V4CLK96L))
$G(\mathbf{X})=\min\begin{cases}3-X_2+\exp(-X_1^2/10)+(X_1/5)^4\\8-X_1X_2&\end{cases}$


---

**quadratic**

“a quadratic function” ([Zhou 等, 2022, p. 6](zotero://select/library/items/WYUCSUFY)) ([pdf](zotero://open-pdf/library/items/I78RP8RT?page=6&annotation=2S5FR2BR)) ***输入3输出1***
$y=-15x_1+2x_1^2-3x_2+x_2^2+5x_3+x_3^2+40$

“a high-dimensional quadratic model” ([Zhou 等, 2022, p. 6](zotero://select/library/items/WYUCSUFY)) ([pdf](zotero://open-pdf/library/items/I78RP8RT?page=6&annotation=JKQ454EG)) ***输入100输出1***
$y=\left(\sum_{i=1}^{100}c_ix_i\right)^2$

“a quadratic performance function” ([Zhao 等, 2024, p. 6](zotero://select/library/items/G7ITCE29)) ([pdf](zotero://open-pdf/library/items/9PPVC2N3?page=6&annotation=TYS3ILMR))
$G(\mathbf{X})=-18X_1+X_2^2+X_2+X_3^2+5X_3+40$

---

> [Piston Simulation Function](https://www.sfu.ca/~ssurjano/piston.html)

“Piston model” ([Constantine和Diaz, 2017, p. 9](zotero://select/library/items/3MX3A9VH)) ([pdf](zotero://open-pdf/library/items/4UNWCK67?page=9&annotation=4JXFDA7E))

“a cylindrical piston model” ([Zhou 等, 2022, p. 7](zotero://select/library/items/WYUCSUFY)) ([pdf](zotero://open-pdf/library/items/I78RP8RT?page=7&annotation=JR6M7PEQ)) ***输入7输出1(t——完成一个周期所用时间)***
用于test parameter screening techniques
$\begin{aligned}&t=2\pi\sqrt{\frac{M}{k+S^2\frac{P_0V_0}{T_0}\frac{T_a}{V^2}}}\\&V=\frac{S}{2k}\left(\sqrt{A^2+4k\frac{P_0V_0}{T_0}T_a}-A\right)\\&A=P_0S+19.62M-\frac{kV_0}{S}\end{aligned}$

“Piston function (PT)” ([Shang 等, 2024, p. 6](zotero://select/library/items/BF4VY3T6)) ([pdf](zotero://open-pdf/library/items/ULWYF55Z?page=6&annotation=QXRTGGLY))
$f_2(\mathbf{x})=2\pi\sqrt{\frac{M_0}{k+Q^2\frac{P_0V_0T_a}{T_0V_a^2}}}$
    

“a cantilever beam” ([Zhao 等, 2024, p. 7](zotero://select/library/items/G7ITCE29)) ([pdf](zotero://open-pdf/library/items/9PPVC2N3?page=7&annotation=FVI5KVV2))
$G(\mathbf{X})=\frac{L}{325}-\frac{\omega bL^4}{8EI}$

“Borehole model” ([Shang 等, 2023, p. 11](zotero://select/library/items/APWQBCLI)) ([pdf](zotero://open-pdf/library/items/4FTIT5GX?page=11&annotation=ABEHA84A)) ***输入8输出1***
高保真度模型：$y_h(\mathbf{x})=\frac{2\pi T_u(H_u-H_l)}{\ln(r_a/r_w)\left[1+\frac{2L_1T_u}{\ln(r_a/r_w)r_w^2K_w}+\frac{T_u}{T_l}\right]}$
低保真度模型：$y_l(\mathbf{x})=\frac{5T_u(H_u-H_l)}{\ln(r_a/r_w)\left[1.5+\frac{2L_1T_u}{\ln(r_a/r_w)r_w^2K_w}+\frac{T_u}{T_l}\right]}$

“Wing weight function” ([Shang 等, 2024, p. 6](zotero://select/library/items/BF4VY3T6)) ([pdf](zotero://open-pdf/library/items/ULWYF55Z?page=6&annotation=6D6MY5A5))
$f_3(\mathbf{x})=\quad0.036x_1^{0.758}x_2^{0.0035}\left(\frac{x_3}{\cos^2(x_4)}\right)^{0.6}x_5^{0.006}x_6^{0.04}\times\left(\frac{100x_7}{\cos(x_4)}\right)^{-0.3}\left(x_8x_9\right)^{0.49}+x_1x_{10}$

“Oakley & O’Hagan Function (OAOH)” ([Shang 等, 2024, p. 6](zotero://select/library/items/BF4VY3T6)) ([pdf](zotero://open-pdf/library/items/ULWYF55Z?page=6&annotation=PUT4R3Y2))
$f_4(\mathbf{x})=\boldsymbol{\xi}_1^\mathrm{T}\mathbf{x}+\boldsymbol{\xi}_2^\mathrm{T}\mathrm{cos}\mathbf{x}+\boldsymbol{\xi}_3^\mathrm{T}\mathrm{sin}\mathbf{x}+\mathbf{x}^\mathrm{T}\varpi\mathbf{x}$


“Stochastic partial differential equation (**PDE**) problem” ([Shang 等, 2023, p. 12](zotero://select/library/items/APWQBCLI)) ([pdf](zotero://open-pdf/library/items/4FTIT5GX?page=12&annotation=RJPAUKK2)) ***输入2(w,v)输出1(y(1,w,v))***
$\frac{d^2y(x,\mathbf{w},\mathbf{v})}{dx^2}-\cos(\pi k_1(x,\mathbf{w}))\frac{dy(x,\mathbf{w},\mathbf{v})}{dx}-\sin(\pi k_2(x,\mathbf{v}))=k_1(x,\mathbf{w})k_2(x,\mathbf{v})$


“Mathematical examples” ([Song 等, 2024, p. 10](zotero://select/library/items/AECRTI4F)) ([pdf](zotero://open-pdf/library/items/BDXU2TUW?page=10&annotation=CFKQ6LMK))

| Function   | p      | Expression                                                                               |
| ---------- | ------ | ---------------------------------------------------------------------------------------- |
| DixonPrice | 30     | $Y = (Xi-1)² - ∑i=1²i(2Xi-Xi+1)², Xi∈[0,10]$                                             |
| Griewank   | 40, 80 | $Y = ∑i=1²iXi/4000 - ∏i=1²icos(Xi/√i) + 1, Xi∈[0,10]$                                    |
| Ackley     | 50     | $Y = -20exp(-0.2√(1/√p)∑i=1²iXi²) - exp(1/√(1/√p)∑i=1²icos(2πXi)) + 20 + e, Xi∈[-10,30]$ |
| Rosenbrock | 70     | $Y = ∑i=1²i[100(Xi+1- Xi)² + (Xi-1)²], Xi∈[0,5]$                                         |
| Morris     | 100    | $Y = α ∑i=1²iXi+β ∑i=1²iXi+Xi, α = √(12-6√(0.1(k1-1)),β = 12√(0.1(k1-1)),Xi∈[0,1]$       |

---

⭐⭐⭐“Ishigami function” ([Shang 等, 2023, p. 9](zotero://select/library/items/APWQBCLI)) ([pdf](zotero://open-pdf/library/items/4FTIT5GX?page=9&annotation=GKWN7TWA)) ***输入3输出1***
高保真度模型：
$y_h(\mathbf{x})=\sin(\pi x_1)+7\sin^2(\pi x_2)+0.1(\pi x_3)^4\sin(\pi x_1)$
由于方法采用多保真度模型，因此还有一个低保真度模型：
$y_l(\mathbf{x})=\sin(\pi x_1)+7.3\sin^2(\pi x_2)+0.08(\pi x_3)^4\sin(\pi x_1)$

“3.2. Ishigami function” ([Palar 等, 2018, p. 182](zotero://select/library/items/3LQAGAB7)) ([pdf](zotero://open-pdf/library/items/P3CAB384?page=8&annotation=HJZGI5K7))
“For the simulation runs we have used p = 4 and (a, b) = (7, 0.1),” ([Antoniadis 等, 2021, p. 10](zotero://select/library/items/XUVZDR4I)) ([pdf](zotero://open-pdf/library/items/BP52YK32?page=10&annotation=DLQGIG94))

“Ishigami function (IS)” ([Shang 等, 2024, p. 6](zotero://select/library/items/BF4VY3T6)) ([pdf](zotero://open-pdf/library/items/ULWYF55Z?page=6&annotation=IDATYQJK))

---

**Reliability**

“A nonlinear strength model” ([Jia和Wu, 2024, p. 8](zotero://select/library/items/866Y2G2M)) ([pdf](zotero://open-pdf/library/items/DZM6NFYY?page=8&annotation=GRHAHS4J)) ***输入6输出1***
性能函数：
$\begin{gathered}g_1=496-\sqrt{\sigma^2+3\tau^2}\\\sigma=\frac{x_5}{x_1(x_4-2x_3)^3/(6x_4)+x_2\left(x_4^3-\left(x_4-2x_3\right)^3\right)/(6x_4)}\\\tau=\frac{x_6}{0.8x_2x_3^2+0.4x_1^3(x_4-2x_3)/x_3}\end{gathered}$

“Nonlinear oscillator structure” ([Jia和Wu, 2024, p. 10](zotero://select/library/items/866Y2G2M)) ([pdf](zotero://open-pdf/library/items/DZM6NFYY?page=10&annotation=DEF22HXT)) ***输入6输出1***

$g_2=3r-\left|\frac{2F_1}{c_1+c_2}\sin\left(\sqrt{\frac{(c_1+c_2)}{2}}\frac{t_1}{m}\right)\right|$


---

“A simple performance function with two random variables” ([Jia和Wu, 2023, p. 10](zotero://select/library/items/PKXKVNP5)) ([pdf](zotero://open-pdf/library/items/3KY8LIBG?page=10&annotation=KNWAFI6F))  ***输入2输出1***
$g_1=\exp\left(0.2x_1+1.4\right)-x_2$

---

### Complex case

“A truss bridge problem” ([Song 等, 2024, p. 14](zotero://select/library/items/I2A9MG56)) ([pdf](zotero://open-pdf/library/items/2TRZV9MT?page=14&annotation=HNVQHJYN)) ***输入120输出201***

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250715154710.png)

桥的尺寸(L=32m、H=5.5m)，卡车过桥的力简化为三个动态随机力$P_{1}(t),P_{2}(t),P_{3}(t)$，建模为stationary Gaussian processes
不确定性参数：crosssectional areas($A_{1}-A_{29}$)，moments of inertia($I_{1}-I_{29}$)，杨氏模量E，泊松比$\nu$，三个力$P_{1}(t),P_{2}(t),P_{3}(t)$
QoI：FEA输出桥中间的Y方向位移，在指定的时间区间$[0,1]$内，区间被离散化为201个节点

---

“A heat conduction problem” ([Song 等, 2024, p. 17](zotero://select/library/items/I2A9MG56)) ([pdf](zotero://open-pdf/library/items/2TRZV9MT?page=17&annotation=VL4TEDBJ)) ***输入57输出51***


![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250715155917.png)

---

“a ten-bar structure” ([Zhou 等, 2022, p. 8](zotero://select/library/items/WYUCSUFY)) ([pdf](zotero://open-pdf/library/items/I78RP8RT?page=8&annotation=G6K8TEDU))  ***输入15输出1***

输出：the vertical displacement at Node 2 as the quantity of interest
![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250716140726.png)

---

“Application case: safety analysis of a radome structur” ([Zhou 等, 2022, p. 9](zotero://select/library/items/WYUCSUFY)) ([pdf](zotero://open-pdf/library/items/I78RP8RT?page=9&annotation=WPC3I9LA)) ***输入17输出1***
输出：the margin of safety (MoS)，如果MoS>0，则复合材料的结构安全
![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250716151913.png)

---

“RAE2822 airfoil problem” ([Shang 等, 2023, p. 13](zotero://select/library/items/APWQBCLI)) ([pdf](zotero://open-pdf/library/items/4FTIT5GX?page=13&annotation=CRU59AIE)) ***输入9输出1(lift coefficient)***

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250716161711.png)
![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250716164437.png)

---

**Reliability**

“A conical structure” ([Jia和Wu, 2024, p. 12](zotero://select/library/items/866Y2G2M)) ([pdf](zotero://open-pdf/library/items/DZM6NFYY?page=12&annotation=EWERZF36)) ***输入9输出1***

$g_3=1-\frac{\sqrt{3(1-\mu^2)}}{\pi Et^2\mathrm{cos}^2\alpha}\left(\frac{P}{2\gamma}+\frac{M}{\lambda r_1}\right)$

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250716193605.png)


“Latch lock mechanism of hatch” ([Jia和Wu, 2024, p. 14](zotero://select/library/items/866Y2G2M)) ([pdf](zotero://open-pdf/library/items/DZM6NFYY?page=14&annotation=L78QBL4P))  ***输入5输出1***

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250716170423.png)
$g_4=r\mathrm{cos}(\alpha_1)+\sqrt{L_1^2-\left(e-r\mathrm{sin}(\alpha_1)\right)^2}+L_2-L_3$

“Riveting process of headless rivet” ([Jia和Wu, 2024, p. 15](zotero://select/library/items/866Y2G2M)) ([pdf](zotero://open-pdf/library/items/DZM6NFYY?page=15&annotation=S8WKE6NM)) ***输入5输出1***

$\sigma_{\max}=K\left(\ln\frac{d^2h-D_0^2t}{2Hd^2}\right)^{nSHE}$
$g_5=\sigma_{sq}-\sigma_{\max}$

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250716172018.png)

“An aero‐engine turbine disk” ([Jia和Wu, 2023, p. 12](zotero://select/library/items/PKXKVNP5)) ([pdf](zotero://open-pdf/library/items/3KY8LIBG?page=12&annotation=HP8TQSWJ)) ***输入6输出1***

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250716172046.png)

$g_2=\sigma_sA-\frac{C(2\pi w)^2}{2\pi}+2\rho(2\pi w)^2H_J$

“Automobile front axle beam” ([Jia和Wu, 2023, p. 14](zotero://select/library/items/PKXKVNP5)) ([pdf](zotero://open-pdf/library/items/3KY8LIBG?page=14&annotation=DZGVLQHX))  ***输入7输出1***

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250716194112.png)
$\begin{aligned}&g_{4}=\sigma_{s}-\sqrt{\sigma^{2}+3\tau^{2}}\\&\sigma=M/W_{x},\tau=T/W_{\rho}\\&W_{x}=\frac{a(h-2t)^{3}}{6h}+\frac{b}{6h}\left[h^{3}-(h-2t)^{3}\right]\\&W_{\rho}=0.8bt^{2}+0.4\left[a^{3}(h-2t)/t\right]\end{aligned}$

“Latch lock mechanism of hatch” ([Jia和Wu, 2023, p. 18](zotero://select/library/items/PKXKVNP5)) ([pdf](zotero://open-pdf/library/items/3KY8LIBG?page=18&annotation=GLF8SXEG)) ***输入5输出1***

$g_5=r\cos\left(\alpha_1\right)+\sqrt{L_1^2-\left(e-r\sin\left(\alpha_1\right)\right)^2}+L_2-L_3$

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250716194347.png)

---


## Open source（Code）

### PyApprox 库

[PyApprox: A software package for sensitivity analysis, Bayesian inference, optimal experimental design, and multi-fidelity uncertainty quantification and surrogate modeling - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1364815223002116#da1) Python

[PyApprox Reference Manual — PyApprox 1.0.3 documentation](https://sandialabs.github.io/pyapprox/)

Methods are available for: 
- **low-rank tensor-decomposition**;
- **Gaussian processes**; 
- **polynomial chaos expansions**; 
- sparse-grids; 
- risk-adverse regression; 
- compressed sensing; 
- Bayesian inference; 
- push-forward based inference; 
- optimal design of computer experiments for interpolation regression and compressed sensing; 
- and risk-adverse optimal experimental design.

[Sensitivity Analysis — PyApprox 1.0.3 documentation](https://sandialabs.github.io/pyapprox/auto_tutorials/analysis/plot_sensitivity_analysis.html#sphx-glr-auto-tutorials-analysis-plot-sensitivity-analysis-py)


### An interactive graphical interface tool

[An interactive graphical interface tool for parameter calibration, sensitivity analysis, uncertainty analysis, and visualization for the Soil and Water Assessment Tool - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1364815222002006#abs0015) 
[tamnva/R-SWAT: This is an interactive web-based app for parallel parameter sensitivity, calibration, and uncertainty analysis with the Soil and Water Assessment Tool (SWAT and SWAT+)](https://github.com/tamnva/R-SWAT) R package

![rswat.gif (1427×793)|333](https://raw.githubusercontent.com/tamnva/R-SWAT/refs/heads/master/inst/R-SWAT/figures/rswat.gif)

### Cluster-based GSA


[Cluster-based GSA: Global sensitivity analysis of models with temporal or spatial outputs using clustering - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S136481522100089X#sec8) R


[sbuis/ClusterBased_GSA: R code for applying the cluster-based GSA method](https://github.com/sbuis/ClusterBased_GSA) 2021




