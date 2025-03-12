---
title: Towards the NASA UQ Challenge 2019——Systematically forward and inverse approaches for uncertainty propagation and quantification
date: 2025-02-27 10:14:54
tags:
  - 
categories: ModelUpdating/Review
---

| Title     | Towards the NASA UQ Challenge 2019——Systematically forward and inverse approaches for uncertainty propagation and quantification                                                                                                   |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Sifeng Bi, Kui He, Yanlin Zhao, David Moens, Michael Beer,  Jingrui Zhang                                                                                                                                                          |
| Conf/Jour | MSSP                                                                                                                                                                                                                               |
| Year      | 2022                                                                                                                                                                                                                               |
| Project   | [Towards the NASA UQ Challenge 2019: Systematically forward and inverse approaches for uncertainty propagation and quantification - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327021007408?via%3Dihub) |
| Paper     |                                                                                                                                                                                                                                    |

<!-- more -->

This work focus on exploring NASA Langley Challenge on Optimization under Uncertainty by proposing a series of approaches for both forward and inverse treatment of uncertainty propagation and quantification. 

# Introduction and problem pre-investigation

The background of the NASA challenge 2019. The overview: 
Subproblems B1) Sensitivity Analysis and C) Reliability Analysis are categorized as **the forward procedure**, 
Subproblems A) Model Calibration, B3) Uncertainty Reduction, D) Reliabilitybased Design, E) Design Tuning, and F) Risk-based Design are categorized as **the inverse procedure**.

> [NASA Langley UQ Challenge on Optimization Under Uncertainty](https://uqtools.larc.nasa.gov/nasa-uq-challenge-problem-2020/)

- Model Calibration & Uncertainty Quantification of Subsystems
- Uncertainty Model Reduction
- Reliability Analysis
- Reliability-Based Design
- Model Updating and Design Parameter Tuning
- Risk-based Design

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250227105511.png)



黑盒模型、输入&输出
- The physical system is represented as a black-box model, i.e. Intergrated system $z(a,e,\theta,t)$， a sub-system $y(a,e,t)$ within the integrated system to derive all sources of uncertainties. 
- The uncertainty of input parameters is categorized as aleatory and epistemic uncertainty. The aleatory uncertainty is presented as variables following probabilistic distributions $a_{i}~f_{a}$. The epistemic uncertainty is presented as unknown-but-fixed constants within predefined intervals $e_{i} \in E$. The distribution and predifined intervals is refered as uncertainty model (UM). Besides the intergrated system takes another design parameter $\theta$ as inputs，and the outputs is further to define the performance function $g(a,e,\theta,z,t)$ and worst-case performance function $\max g_i (a,e,\theta)$
- The output is time-domain sequence. 

问题：since only a wide boundary [0, 2] is given, 并且缺乏参数的分布信息（分布形式&系数），先验知识是非常有限的。主任务之一是根据给定的观察时间序列$D_{1}$来优化UM （inverse: A, B3, E2）

(forward:B1,c) 由于参数包含aleatory and epistemic uncertainty，因此被表示为P-box，
任务的目标是：1.研究如何将两不确定性从输入传播到输出，2.不确定性如何影响系统的故障概率。
任务需要：1.将两不确定性进行解耦decoupling的方法，2. 测量时序数据P-box的完全量化方法

(inverse:D, E4, F) 需要根据提供的baseline design 寻找一种 new design，将系统的可靠性集成到目标函数中。trade-off：认知不确定性忽视多少 与 从中获取的收益 Risk & Gain $\theta_{r\%risk} \ell(r)$
关键挑战：1.the definition of the optimal criterion, 2. the development of the computational viable optimization algorithm.

# Theoretical development

## Forward decoupling approach for uncertainty propagation

### Parameterization hypothesis of the input P-box

UM:
- the e is the interval of unknown-but-fiex parameter
- the **Beta Mixture Model** (BMM) to represent the UM of a， reason：
  - 1) the format flexibility 可以表示多种形式PDF
  - 2) the interval definiteness 挑战的参数设置在在确定的区间内，通过调整权重参数来控制区间

The basic Beta distribution is a continuous probability distribution defined on the interval $[0, 1]$ with two positively real shape coefficients A and B：
$\mathrm{Beta}(x,A,B)=\frac{\Gamma(A+B)}{\Gamma(A)\Gamma(B)}x^{A-1}(1-x)^{B-1}$, where the $\Gamma(\cdot)$ is the gamma functon $\Gamma()=\int_0^\infty x^{-1}e^{-x}dx$

The BMM is defined as the sum of N basic Beta distributions：$\mathbf{BMM}(\mathbf{x})=\sum_{i=1}^N\beta_i\mathbf{Beta}(x,A_i,B_i)$，$\beta$为权重系数，$\sum_{i=1}^N\beta_i=1$

[BMM(Beta Mixture Model) | Desmos](https://www.desmos.com/calculator/ggohybbhhq?lang=zh-CN)

$f_{a_i}\left(a,A_1^{(i)},B_1^{(i)},A_2^{(i)},B_2^{(i)}\right)=\frac{1}{4}\mathrm{Beta}\left(\frac{1}{2}a,A_1^{(i)},B_1^{(i)}\right)+\frac{1}{4}\mathrm{Beta}\left(\frac{1}{2}a,A_2^{(i)},B_2^{(i)}\right)$ 通过两个Beta函数融合来获得区间在$[0,2]$ 内的BMM, 为什么选两个? reason:
- the BMM with only one component is reduced to the normal Beta distribution and hence not flexible enough to fit the potential true distribution
- the BMM with too many components will lead to too much unknown coefficients leading the updating process very difficult
- As a result, the choice of the twocomponent BMM has a balance of the flexibility and complexity

### The two-loop approach for uncertainty propagation in the form of P-box

由于输入包含了aleatory 和 epistemic 两种不确定性，因此输出也包含这两种，被特征为imprecise probabilities i.e. P-box

| This work                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------- |
| ![image.png\|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250227161029.png) |
| Stochastic Model Updating with Uncertainty Quantification_An Overview and Tutorial                                |
| ![image.png\|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250224110038.png) |
|                                                                                                                   |

The proposed strategy makes it possible for the **forward reliability analysis** to evaluate the range of the system failure probability when the epistemic uncertainty is involved.

## Feature extraction and uncertainty quantification for time-domain variables

从时域数据中提取特征，以便UQ

### The Empirical Mode Decomposition

EMD 也称作 Hilbert-Huang transform. EMD decompose the time-domain signal to Intrinsic Mode Functions (IMF), which following two constraints
- An IMF is a function run through the horizontal axis multiple times with **the number of poles and the number of zero crossings to the axis** must be either equal or be differ at most by one. IMF通过水平轴多次，并且极点和零点的数量必须相等或者最多相差1
- **the envelop** defined by the multiple maximums and minimums **should be symmetric according to the horizontal axis.** 最大和最小值的包络线必须关于x轴对称

IMF的这两个特性使得其可以很容易和全面地提取到时序数据地不确定性特征，然后可以用之前通用的UQ指标进行计算

### The Bhattacharyya distance: A comprehensive UQ metrics

$d_B(y_{obs},y_{sim})=-log\left\{\int_Y\sqrt{P(y_{obs})P(y_{sim})}dy\right\}$

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250228132959.png)

## Inverse uncertainty reduction and optimization
### Bayesian model updating

$P(x|y_{obs})=\frac{P_L(y_{obs}|x)P(x)}{P(y_{obs})}$
- $P_{L}(y_{obs}|x)\infty exp\left\{-\frac{d^{2}}{\varepsilon^{2}}\right\}$
- TMCMC sampling algorithm

### Optimization tools selection

为了实现reliability-based and risk-based task，要求：
- 优化必须在高维空间中进行，design parameter $\theta$是一个没有给定边界的9维变量
- 优化标准仅仅被基于P-box定义，time-consuming

The Particle Swarm Optimization (PSO) method 优点：高速收敛，高维空间非线性函数的适应性

For the Subproblem F) Risk-based design requires to define the Risk to represent the portion of the epistemic uncertainty to be neglected, and the corresponding Gain resulting from taking the risk. --> 导致优化变得复杂且昂贵 --> surrogate optimization algorithm

PSO+surrogate model

# Problem investigation and outcome analysis
## Subproblem A): Model calibration & UQ for the subsystem  

### Stochastic model updating process

- 参数化
- 特征提取
- 参数修正

total 24 parameters to be update：(Parameterization)
- 5个aleatory variables：BMM with 4 parameters $A_{1}^{(i)},B_{1}^{(i)},A_{2}^{(i)},B_{2}^{(i)}$
- 4个epistemic variables：unknown-but-fixed constant within pre-defined interval

In Bayesian Model Updating：(Parameter calibration)
- prior distribution： uniform for 24 parameters
- EMD method and Bhattacharyya distance are employed to quantify the discrepancy (Feature extraction)
  - Select featured time poitns from IMF calculated from EMD 没有统一标准，通用做法是覆盖整个时间序列上的局部最大和最小值
- posterior distribution
  - aleatory parameters $(A_1^{(i)},B_1^{(i)},A_2^{(i)},B_2^{(i)},i=1,\cdotp\cdotp\cdotp,5)$, 用PDF的最大值极点，表示最大似然
  - epistemic parameters $(e_i,i=1,\cdotp\cdotp\cdotp,4)$，使用Kernel Density Estimation (KDE) approach 来根据poseterior samples估计后验PDFs，然后归一化得到normalized PDF。由于希望获得一个最大的reduced interval of e variables仍然可以包含真实的e，因此选取$\alpha=0.1$处的截断区间

### Assessment of the UM calibration results

1. 对比目标、原始(通过随机在原始区间$[0,2]$上采样)和修正后的时序数据 through $y(a,e,t)$
2. 对比特殊的time points of sequence(直接在time-domain sequence上采样，而不用IMF)，对比special time points的分布
3. 定量比较特殊时间点的BD值，并且探讨了当目标数据$D_{1}$的样本量减少时对修正结果的影响

## Subproblem B): Uncertainty reduction

- 敏感性分析参数e对输出y的影响
- 回复host 对基于敏感性分析结果进行uncertainty reduction的请求,得到refined UM with reduced interval $E_{1}$
- 重复子问题A的修正，进一步修正输入变量的UM，以便于 修正后的 $e\sim E\subseteq E_1$

ranking criterion定义：由于output feature 中包含认知和随机不确定性，并且是时序数据，因此使用序列数据沿着时域的degree of dispersion 来作为 criterion of the influence of epistemic uncertainty，i.e. $D(t)=\max_ny(t)-\min_ny(t)$ ，其中n为序列数据的样本数，换句话说，$D(t)$是所有样本time sequence的包络线的区间大小

origin $D_{0}(t)$为初始包络(e从初始区间$[0,2]$上采样10000个样本，a从 calibrated 分布中采样10000个样本)，然后e variables 的敏感性被定义为从$D_{0}(t)$减少的程度： $\Delta D_i(t)=D_0(t)-D_i(t)0s\leq t\leq5s$，$D_{i}$为只updated 第i个e的包络(posterior distribution的最大值 $\alpha=1.0$)，其他e variable 固定

1. 对比包络线和$\Delta D_i(t)$
2. 定量对比5s内的平均$\Delta D_i(t)$，越大越敏感

基于敏感性分析结果，进一步 reduce uncertainty，再次使用Bayesian udpating

1. 对比进一步修正后 variables 的pdf
2. 定量比较进一步修正的参数，以及BD of special time points from output time-domain sequence
3. 对比special time points的分布

证明通过进一步的updating，可以提高修正结果的精度 or the significance of the interval reduction to the calibration effect

## Subproblem C): Reliability analysis of baseline design

the integrated model $z(a,e,\theta,t)$，并定义了一系列performance functions：
- $g_{1}$是问题提供的内置函数
- $g_{2}=\max_{t\in\left[\frac{T}{2},T\right]}|z_{1}(a,e,\theta,t)|-0.02$
- $g_{3}=\max_{t\in[0,T]}|z_{2}(a,e,\theta,t)|-4$

worst-case performance function：$w(a,e,\theta)=\max_{i=1,2,3}g_i(a,e,\theta)$ （特性：任何一个performance function 大于0，则该指标大于0）

通过三个指标进行可靠性分析：**根据需求/性能计算故障概率，或者抛弃需求时的严重程度**
- the failure probability for each individual requirement $g_{i}$, i.e. $R_i(\theta)=\left[\min_{e\in E}\mathbb{P}[g_i(a,e,\theta)\geq0],\max_{e\in E}\mathbb{P}[g_i(a,e,\theta)\geq0]\right]$
  - 理解：故障概率$\mathbb{P}$，通过性能函数大于等于0判断，以$g_2$为例，也就是当output features在$\left( \frac{T}{2},T \right)$ 内绝对值的最大值大于等于0.02时，认为integrated model 系统故障
- the failure probability for all requirements $g_{1-3}$, i.e.  $R(\theta)=\left[\min_{e\in E}\mathbb{P}[w(a,e,\theta)\geq0],\max_{e\in E}\mathbb{P}[w(a,e,\theta)\geq0]\right]$
- the severity of each individual requirement violation $g_{i}$, i.e. $S_i=\max_{e\in E}\mathbb{E}[g_i|g_i\geq0]P[g_i\geq0]$
  - $\mathbb{P}[\bullet]\text{ is the probability operator, }\mathbb{E}[\bullet|\bullet]\text{ is the conditional expectation}$

1. 通过double-loop计算 $R_{i},R,S_{i}$
2. 固定其他参数，只改变其中一个e variable来对比 range of $R_{i},R$的变化程度

## Subproblem D): Reliability-based design

需要一个优化指标来表示可靠度(基于worst-case failure probability)，为了解决C问题指标用double-loop的计算耗时，提出一个简化指标：
最小化该指标(故障概率的最大值)：$f(\theta)=\max_{e\in E,a}\mathbb{P}[w(a,e,\theta)\geq0]$ ，代替double-loop，而是只采样一次，进行single-loop(共采样M个e，对每个随机采样的e，随机采样N个a，故障率通过N个a进行计算，然后根据得到的M个e对应的M个故障率，最后求的最大值即为指标)

PSO算法通过最小化故障率来寻找对应的design parameter，需要设定一个搜索区间，通常是$\theta_{basline}$的$\pm 50\%$

寻找到最优的$\theta_{new}$后，
1. 与C子问题类似，计算对应的R和S区间，并于之前$\theta_{basline}$的结果进行对比。
2. 输出特征对比$\theta_{basline}$ 和 $\theta_{new}$

## Subproblem E): Model update and design tuning

Task：
- 使用$\theta_{new}$得到新的观察集$D_{2}$ ,并使用新数据集利用 sub_A ($D_{1}$ of $y(a,e,t)$) 进一步修正UM (该子问题 利用 $D_{2}$ of $z(a,e,\theta,t)$ 修正UM)
- 基于修正的UM，进一步请求 e variable的refinements，重复 sub_A and sub_D 修正UM，并优化design parameters$\theta_{final}$
- 重复 sub_C 的可靠性分析，并比较可靠性指标 of $\theta_{basline}, \theta_{new},\theta_{final}$

用EMD提取$z_{1},z_{2}$的特征，使用Bayesian model udpating with TMCMC sampling algorithm 对UM进行修正

1. 对比使用$D_{2}$修正后的 UM 与 sub_A中使用$D_{1}$修正后的UM区别

接着重复 sub_B，进一步通过model updating 得到 $UM_{final}$

1. 对比进一步修正$UM_{final}$，与一次修正的区别
2. 对比在$\theta_{new}$下，$D_{2}$与使用$UM_{final}$生成的$z_{1},z_{2}$的区别
3. 对比可靠性指标$R_{i},R,S_{i}$ of $\theta_{basline}, \theta_{new},\theta_{final}$

## Subproblem F): Risk-based design

考虑从E问题中得到的remaining epistemic uncertainty space，risk r%被定义为epistemic space E 被忽视的部分，$r \in [0,100]$, gain被定义为系统性能的改进 benefitting from retained 100-r% epistemic space。忽视的越多，风险r%越高，同时gain(在保留的epistemic uncertancy space系统可靠性的改进程度)应该获取更多

the gain： $l(r)=\frac{P_E}{P_{E^*}}=\frac{\max_{e\in E}\mathbb{P}[w(a,e,\theta)\geq0]}{\max_{e\in E^*(r)}\mathbb{P}[w(a,e,\theta)\geq0]}$，可以理解为忽视worst-case最大故障率的e区间后，故障率减小的比例有多少，减小的越多收益越高

epistemic uncertainty space忽视部分：$E_{cut}(r,e^\star)=\begin{bmatrix}e^\star\pm0.5•(e_{upper}-e_{lower})•r\%\end{bmatrix}$
- the upper bound and lower bound of the epistemic interval $e_{upper},e_{lower}$
- $e^*=\underset{e\in E}{\operatorname*{\operatorname*{argmax}}}\mathbb{P}[w(a,e,\theta)\geq0]$, 可以推断出 the maximum failure probability in the full space $P_{E}$ is larger than the one evaluated in the retained space $P_{E^*}$,因此 the gain $l>1$，可以理解为，忽视的部分是对worst-case最大故障率的 e 区间

The retained epistemic space: $E^*(r)=E-E_{\mathrm{cut}}(r,e^*)$
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250302182427.png)

risk-based design: 寻找一个design point $\theta_{r\%risk}$，使得$l(r)$最大，使用PSO for Sub-optimization，为了减小 the outer loop 计算 gain的计算负担， 使用surrogate optimization algorithm (adaptive Radial Basis Function，RBF)来代替复杂的有限元计算。

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250302183146.png)

1. 对比r=0.05, $\theta_{final}$与$\theta_{\hat{r}\%risk}$在full space 和 retained space 上的可靠性指标，retained 对比full，可靠性指标都有所提升。相较于$\theta_{final}$，$\theta_{\hat{r}\%risk}$ 的可靠性指标性能较差，因为其是局部的优化，而$\theta_{final}$是通过full space 上全局优化得到的

研究risk 与 gain之间的trade-off，得到不同risk level $[0,0.05,0.5,1,5,10]$下最优的设计参数$\theta_{r\%risk}$，并评估对应的gain

1. 对比不同risk level下，the gain of $\theta_{r\%risk}$比 $\theta_{final}$更大，且the gain随着risk level增加而增加
2. 通过绘图对比，trade-off，尽可能在risk 增加时，更多地获取gain


# Conclusion

