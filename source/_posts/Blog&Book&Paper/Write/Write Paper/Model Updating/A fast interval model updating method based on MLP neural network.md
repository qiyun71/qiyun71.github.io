---
title: A fast interval model updating method based on MLP neural network
date: 2024-03-13 16:52:45
tags: 
categories:
---

| Author | m202210465@xs.ustb.edu.cn \| zhaoyanlin@ustb.edu.cn \| ydb@ustb.edu.cn |
| ------ | ---------------------------------------------------------------------- |

<!-- more -->

| 区别        | EI 会议                   | SCI                                                                                                    |
| --------- | ----------------------- | ------------------------------------------------------------------------------------------------------ |
| MLP-based | 动力学响应-->结构参数，单次对单次      | 动力学响应-->结构参数，区间对区间                                                                                     |
| MLP       | Inverse surrogate model | Interval identification model                                                                          |
| 算例        | 三自由度弹簧、钢板               | 三自由度弹簧、钢板、飞机                                                                                           |
| 期刊        | ISRERM                  | [Chinese Journal of Aeronautics](https://www.sciencedirect.com/journal/chinese-journal-of-aeronautics) |


Finite Element (FE)
Monte Carlo (MC)
Interval Response Surface Models (IRSM)
Radial Basis Function (RBF)
Interval Overlap Ratio (IOR)
Interval Deviation Degree (IDD)
Back Propagation (BP)
Multi-Layer Perceptron (MLP)
Sparrow Search Algorithm (SSA)

# EI

## Abstract

区间模型更新被广泛用于结构系统知识不足的情况。传统的区间模型更新方法依靠优化算法来更新不确定参数的区间边界。然而，这种方法存在优化时间长、无法为高维输出特性确定合适的不确定性量化指标等局限性。因此，我们提出了一种基于 MLP（多层感知器）神经网络的快速区间模型更新框架，它将已知的模型特征作为输入来预测相应的结构参数。该框架建立了一个基于 MLP 的反演代理模型，将传统方法的反演问题转化为正演问题进行解决。利用广泛的模型特征和结构参数对，通过正向计算和反向传播，最终拟合出精确的反代模型。网络训练只需要构建简单的度量指标，训练完成后的网络校正速度非常快。两个经典的数值实例--质量弹簧系统和钢板结构--证明了本文所介绍方法的可行性和有效性。

- Introduction
- Basic theory 
  - Uncertainty propagation
  - MLP neural network
- Interval model updating based on MLP procedure
- Numerical case studies: a mass-spring system
- Experimental case study: steel plate structures 

## 公式

$\mathbf{X}^I = [\underline{\mathbf{X}},\overline{\mathbf{X}}] = [\mathbf{X}_c - \Delta \mathbf{X},\mathbf{X}_c + \Delta \mathbf{X}]$

$$\mathbf{Y}_{sim}^{c}=\frac{1}{2}\Bigg(\max\limits_{1\leq i\leq N_s}\mathbf{Y}_{sim}^i+\min\limits_{1\leq i\leq N_s}\mathbf{Y}_{sim}^i\Bigg)$$
$$\mathbf{Y}_{exp}^{c}=\frac{1}{2}\Bigg(\max\limits_{1\leq j\leq N_e}\mathbf{Y}_{exp}^j+\min\limits_{1\leq j\leq N_e}\mathbf{Y}_{exp}^j\Bigg)$$
$$\Delta\mathbf{Y}_{sim}=\frac{1}{2}\Bigg(\max_{1\leq i\leq N_{s}}\mathbf{Y}_{sim}^i-\min_{1\leq i\leq N_{s}}\mathbf{Y}_{sim}^i\Bigg)$$
$$\Delta\mathbf{Y}_{exp}=\frac{1}{2}\Bigg(\max_{1\leq j\leq N_{e}}\mathbf{Y}_{exp}^j-\min_{1\leq j\leq N_{e}}\mathbf{Y}_{exp}^j\Bigg)$$
$\mathbf{Y}_{sim}^\mathrm{I}=[\underline{\mathbf{Y}_{sim}},\overline{\mathbf{Y}_{sim}}]=[\mathbf{Y}_{sim}^c-\Delta\mathbf{Y}_{sim},\mathbf{Y}_{sim}^c+\Delta\mathbf{Y}_{sim}]$
$\mathbf{Y}_{exp}^\mathrm{I}=[\underline{\mathbf{Y}_{exp}},\overline{\mathbf{Y}_{exp}}]=[\mathbf{Y}_{exp}^c-\Delta\mathbf{Y}_{exp},\mathbf{Y}_{exp}^c+\Delta\mathbf{Y}_{exp}]$

$\mathcal{F}_{\mathbf{\Theta}}^{\prime}$

$n^H_{11} =\sigma( \sum_{l=1}^{l=i} w^{IH_1}_{l1} \times n^{I}_l + b_1^{H_1})$

$\mathbf{n}^O = \mathcal{F}_M(\mathbf{n}^I)$

$\widehat{\mathbf{X}}^{j}_{exp}(j=1,2,\cdots,N_{e})$

$\mathrm{IDD}(\mathbf{Y}^\mathrm{I}_{sim}\mid \mathbf{Y}^\mathrm{I}_{exp})=\frac{\mathrm{len}((\mathbf{Y}^\mathrm{I}_{sim}\cup \mathbf{Y}^\mathrm{I}_{exp})-(\mathbf{Y}^\mathrm{I}_{sim}\cap \mathbf{Y}^\mathrm{I}_{exp}))}{\mathrm{len}(\mathbf{Y}^\mathrm{I}_{exp})},$


$w^{IJ}_{ij}=w^{IJ}_{ij}-\eta\frac{\partial{L_1}}{\partial w^{IJ}_{ij}}$


$\begin{aligned}&m_{1}=1\left(\mathrm{kg}\right),m_{2}=1\left(\mathrm{kg}\right),m_{3}=1\left(\mathrm{kg}\right);\\&k_{3}=k_{4}=1\left(\mathrm{N/m}\right),k_{6}=3\left(\mathrm{N/m}\right);\\&k_{1}=k_{2}=k_{5}=[0.8,1.2](\mathrm{N/m})\end{aligned}$

$\omega_1^2,\omega_2^2,\omega_3^2$
$|\varphi(1,1)|$


## References

[Structural Dynamic Model Updating Techniques: A State of the Art Review | Archives of Computational Methods in Engineering](https://link.springer.com/article/10.1007/S11831-015-9150-3#Fig12)
[Finite element model updating taking into account the uncertainty on the modal parameters estimates - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0022460X06002719)

Deterministic
[Deterministic and probabilistic-based model updating of aging steel bridges - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352012423006239#b0005)


1. [Model Updating In Structural Dynamics: A Survey - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022460X83713404)
2. [Sensitivity-based finite element model updating of a pontoon bridge - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0141029616307830)
3. [A novel sensitivity-based finite element model updating and damage detection using time domain response - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0022460X22003832)
4. [The sensitivity method in finite element model updating: A tutorial - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327010003316)
5. [Finite element model updating using the shadow hybrid Monte Carlo technique - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327014002301)
6. [A finite element model updating method based on the trust region and adaptive surrogate model - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0022460X23001505)
7. [Sequential surrogate modeling for efficient finite element model updating - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S004579491630027X)
8. [Dealing with uncertainty in model updating for damage assessment: A review - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327014004130)
9. [Stochastic Model Updating with Uncertainty Quantification: An Overview and Tutorial - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327023006921#s0015)
10. [Bayesian Updating: Reducing Epistemic Uncertainty in Hysteretic Degradation Behavior of Steel Tubular Structures | ASCE-ASME Journal of Risk and Uncertainty in Engineering Systems, Part A: Civil Engineering | Vol 8, No 3](https://ascelibrary.org/doi/10.1061/AJRUA6.0001255)
11. [Stochastic Bayesian Model Updating to Reduce Epistemic Uncertainty in Satellite Attitude Propagation | AIAA SciTech Forum](https://arc.aiaa.org/doi/10.2514/6.2024-0200)
12. [A survey of non-probabilistic uncertainty treatment in finite element analysis - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0045782504004049)
13. [An interval-based technique for FE model updating](https://www.inderscience.com/offers.php?id=26836)
14. [Interval model updating with irreducible uncertainty using the Kriging predictor](Interval%20model%20updating%20with%20irreducible%20uncertainty%20using%20the%20Kriging%20predictor.md) +质量弹簧算例参考
15. [An interval model updating strategy using interval response surface models](An%20interval%20model%20updating%20strategy%20using%20interval%20response%20surface%20models.md) +质量弹簧对比方法 1
16. [Interval model updating using perturbation method and Radial Basis Function neural networks](Interval%20model%20updating%20using%20perturbation%20method%20and%20Radial%20Basis%20Function%20neural%20networks.md) +Introduction 倒数第二段 RBF 神经网络
17. [Efficient inner-outer decoupling scheme for non-probabilistic model updating with high dimensional model representation and Chebyshev approximation - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327022011086)
18. [Analysis of Uncertain Structural Systems Using Interval Analysis | AIAA Journal](https://arc.aiaa.org/doi/10.2514/2.164) 
19. [Modified perturbation method for eigenvalues of structure with interval parameters | Science China Physics, Mechanics & Astronomy](https://link.springer.com/article/10.1007/s11433-013-5328-6)
20. [Interval parameter sensitivity analysis based on interval perturbation propagation and interval similarity operator - Archive ouverte HAL](https://hal.science/hal-04273667)
21. [An improved interval model updating method via adaptive Kriging models | Applied Mathematics and Mechanics](https://link.springer.com/article/10.1007/s10483-024-3093-7)
22. [An Interval Model Updating Method Based on Meta-Model and Response Surface Reconstruction | International Journal of Structural Stability and Dynamics](https://www.worldscientific.com/doi/10.1142/S0219455423501158)
23. [A PCA-Based Approach for Structural Dynamics Model Updating with Interval Uncertainty | Acta Mechanica Solida Sinica](https://link.springer.com/article/10.1007/s10338-018-0064-0)
24. [Interval identification of structural parameters using interval overlap ratio and Monte Carlo simulation - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S096599781731164X) +质量弹簧对比方法 2 + IOR 量化指标
25. [Interval Identification of Structural Parameters Using Interval Deviation Degree and Monte Carlo Simulation | International Journal of Computational Methods](https://www.worldscientific.com/doi/abs/10.1142/S0219876218501037) + 质量弹簧 RSM 参考 +质量弹簧对比方法 3 + IDD 量化指标
26. [The sub-interval similarity: A general uncertainty quantification metric for both stochastic and interval model updating - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327022004575?via%3Dihub)
27. [(PDF) Stochastic model updating based on sub-interval similarity and BP neural network](https://www.researchgate.net/publication/367239633_Stochastic_model_updating_based_on_sub-interval_similarity_and_BP_neural_network) +第二章 SSA 区间模型修正算法
28. [Model updating of an existing bridge with high-dimensional variables using modified particle swarm optimization and ambient excitation data - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S026322412030292X)
29. [Bridge Model Updating Using Response Surface Method and Genetic Algorithm | Journal of Bridge Engineering | Vol 15, No 5](https://ascelibrary.org/doi/10.1061/%28ASCE%29BE.1943-5592.0000092)
30. [Efficient Bayesian inference for finite element model updating with surrogate modeling techniques | Journal of Civil Structural Health Monitoring](https://link.springer.com/article/10.1007/s13349-024-00768-y)


- [x] 修改图片
- [x] 摘要
- [x] 添加相关文献
- [x] 结论
  - [x] 首先在所选结构参数的合适区间内生成一定量的数据，并根据有限元仿真计算得到对应的响应特征数据，两者共同组成训练集来训练 MLP 神经网络。训练后的神经网络即有限元逆代理模型可以根据响应特征准确地预测出对应的结构参数且推理速度非常快，其可以用于实时区间模型修正。
  - [x] 从三自由度弹簧和钢板两个算例可以看出，训练后的 MLP 可以准确地拟合有限元逆向计算的过程，训练后的 MLP 可以用于快速高精度的模型修正。

# SCI

### ***New IDEA？***

这篇SCI的~~Self-supervised~~ Response-consistent Supervised Interval Model Calibration方法，假定把里面的神经网络叫做**自监督区间校准模型SICM**，它可以有两种训练的方法： (**目前我现在论文使用的是第一种，之前我在introduction有个创新点想错了——加快训练速度/减少训练次数，实际上SICM并没有这个能力**)
1. 如果用生成的(大量)仿真区间数据$\boldsymbol{Y}^{\mathbf{I}}_{sim}$进行训练，那训练结束后可以做到快速的校准/辨识，就是训练速度很慢(数据集大)。这种方法意义不是很大，只能说相较于传统Model Updating方法有速度提升。实际上与Naive Interval Model Calibration (NICM)相比没有改进，硬凑的一点是能将model validation的精度提升，也就是减小校准后$\boldsymbol{Y}^{\mathbf{I}}_{cal}$与实验的$\boldsymbol{Y}^{\mathbf{I}}_{\exp}$之间的平均误差，但是这里有一个大前提是SICM校准后model validation使用的区间传播 (Interval propagation) 方法必须与SICM结构中用的区间传播方法相同。 **!!!!!!** 如果所有(即训练集生成、SICM结构、model validation)的区间传播用同一种方法，那么SICM与NICM相比，没有任何(精度/效率)提升，这是由于训练集生成时通过区间传播生成，训练集中的$\boldsymbol{X}^{\mathbf{I}}$ 与 $\boldsymbol{Y}^{\mathbf{I}}$ 是相对应的，因此SICM监督$\boldsymbol{X}^{\mathbf{I}}$ 与 监督 $\boldsymbol{Y}^{\mathbf{I}}$ 之间的效果是一样的。**| SICM方法创新点：1. 与传统方法相比可以快速辨识区间参数  2. (硬凑的点)当训练集生成与model validation的区间传播方法不同，且当SICM结构中区间传播方法与model validation中方法相同时，SICM相较于NICM来说model validation时有精度提升。*也就是说创新点为SICM可以消除数据集生成时由于区间传播方法所带来的误差，提升model validation的精度(相较于之前的方法BCNN/DCNN都没有考虑区间模型修正中区间传播带来的误差)* |** (实际上，所有区间传播使用同一种方法，精度会更高一点)
2. 如果用(一组)实验数据$\boldsymbol{Y}^{\mathbf{I}}_{\exp}$进行训练，神经网络训练很快而且网络收敛后也可以做到快速的校准/辨识。**这是一种在线训练的方式，有点类似Updating的过程，但又有所差异**：
    - 传统的Model Updating：SSA等优化算法不断寻找最优结构参数$\boldsymbol{X}^{\mathbf{I}}_{optimal}$，让$\boldsymbol{Y}^{\mathbf{I}}_{sim}$与$\boldsymbol{Y}^{\mathbf{I}}_{\exp}$之间的差异越来越小
    - SICM进行 Updating：通过loss函数(网络预测的$\boldsymbol{Y}^{\mathbf{I}}_{predict}$与$\boldsymbol{Y}^{\mathbf{I}}_{\exp}$之间的差异)，反向传播不断调整神经网络的参数(权重参数和偏差参数)，最终可以让网络输出符合要求的辨识参数$\boldsymbol{X}^{\mathbf{I}}_{predict}$。**训练后的MLP就拥有了Model Calibration的能力，但是理论上只能准确地Calibrate 这一组实验数据**

方法1(目前这篇论文方法) 与 方法2 的精度对比：

| $\boldsymbol{Y}^{\mathbf{I}}_{predict}$于$\boldsymbol{Y}^{\mathbf{I}}_{\exp}$的平均误差 | 1. SICM训练 (Identification) | 2. SICM 在线训练 (Updating) |
| --------------------------------------------------------------------------------- | -------------------------- | ----------------------- |
| 区间下界                                                                              | 0.326                      | 0.420                   |
| 区间上界                                                                              | 0.307                      | 0.295                   |
| 训练时间                                                                              | 60min以上                    | 不到1min                  |
| 推理时间                                                                              | 1s                         | 1s                      |

目前的对这两种方法的想法：
- 方法一网络是通过大量的有限元仿真数据训练死的，只是一个有限元参数 逆辨识模型/校准模型。(更适合学术)。意义不大，除非生成数据集的时候不进行区间传播(不先生成结构参数区间$\boldsymbol{X}^{\mathbf{I}}$，然后区间传播得到$\boldsymbol{Y}^{\mathbf{I}}$ )，而是只随机生成一些特征频率区间$\boldsymbol{Y}^{\mathbf{I}}$，但这样由于这些随机生成的特征频率区间没法通过SICM神经网络预测+区间传播得到，可能会导致网络无法收敛。所以还不如方法二——直接使用实验数据来进行训练。
- 方法二网络是活的，通过测量实验数据和在线训练的方式可以不断增强网络的能力。(更适合工程)。但是意义也不大，因为一组/一次实验的数据只需要校准/辨识一次，除非多次测量的实验区间参数非常相近，这样网络收敛的速度会很快。

(区间模型修正中) 实验测量特征频率的区间，应该会有多次测量的数据，而不是只有这样一组实验数据：

| 特征频率  | 不确定性区间参数(测量)     |
| ----- | ---------------- |
| $f_1$ | [42.66, 43.64]   |
| $f_2$ | [118.29, 121.03] |
| $f_3$ | [133.24, 136.54] |
| $f_4$ | [234.07, 239.20] |
| $f_5$ | [274.29, 280.64] |

***但是实验测量数据是很难获得的，所以方法2在线训练可以通过以下流程让网络逐渐增强：***
- 先在第一组实验区间数据上进行训练，训练快速收敛后停止训练，网络参数固定，此时网络可以准确校准这一组实验的区间
- 然后当第二组实验被测得时，与一组实验区间数据组合，对网络进行继续训练，同样收敛后停止，此时理论上网络可以准确校准这两组实验的区间
- 依次类推，随着实验次数增多，训练时间会不断加长，但是推理也就是校准/辨识的时间还是很快

**实验数据越多，网络训练越多，神经网络校准能力越强**

### IDEA

***无法解决的问题***：
- [ ] 模型验证的精度并不高，这是由于：有限元模型通过只修改几个结构参数，无法仿真出符合实验测量的动力学响应特征(实验测量过程中有很多的误差，无法量化)
- [ ] 自监督的方法...没实际意义，除非 **数据集只生成动力学响应，而不通过结构参数生成** (***但是这样精度很低***，不如直接使用实验测量数据进行训练)

#### IDEA 思考过程

响应区间修参数区间
- 首先基于 MLP 构建代理模型
- 使用 MLP SM 生成区间数据~~(要不要结合 I-box 的思路，基于 I-box 构建训练集增强网络鲁棒性)~~
- 训练 MLP 神经网络对区间量进行修正

算例：
- 三自由度弹簧，well-separated mode
- 钢板算例
- 飞机算例

**区间预测区间如何构建数据集**？(重要)
- 在一定($a\in[290,310]$)范围内随机生成n个lower，对于每个lower生成n个upper(相当于生成了n个区间长度)，然后得到了 $n^{2}$ 组 lower 和 upper
- **小区间是否有必要，一直预测不准** *太小了没必要，当作误差处理*

- 调查一些 MLP 改进的文章，参考如何写作
- 针对小区间预测不准确的额问题，使用一个小算例验证一下

在生成区间数据的时候使用了 MC 方法，在结构参数的区间内随机采样，然后进行 FEA 分析得到的动力学响应的区间有一定的误差(结构参数的 lower bound 不一定对应动力学响应的 lower bound，因此可能没有采样到动力学响应 bound 对应的那两组结构参数)

目前想法：**MC 采样的 UP 必然会导致误差，具体体现为计算出来的响应区间相对实际会变小一些**，同时使用 FE 代理模型也会导致一部分的误差
- 分层采样：数据集生成的 MC 采样部分，使用划分区间式(子区间)的采样方法，(原来是在 Interval 上进行 N 组采样，现在将采样点数量划分为 n 小组，每小组有 N/n 个采样点，然后根据每个小组进行有限元计算得到模态频率的 sub interval，n 个小组得到 n 个模态频率子区间，然后求这 n 个小组的最大边界)。**分层均匀采样**
  - 从区间中采样必然会导致误差，**采样的点中不包含参数的上下边界两个值**。改进：如果要从区间中采样 1000 个点，何不只采样 998(=1000-2)个点，然后加上两个边界值
  - LHS 采样方法使用的是**随机配对 Pair randomly**的方法：经过随机配对，也可能会导致有些部分未被采样，精度最高的的还是**完全不考虑维度爆炸**。
- ~~MLP 网络的输出添加一些噪声(MC 采样导致区间的误差噪声、有限元代理模型导致的误差噪声)~~ 本质是想要消除神经网络作代理模型带来的误差(很难消除)
  - ~~[Monte Carlo Study of the Effect of Measurement Noise in Model Updating with Regularization]( https://www.sci-hub.wf/10.1061/ (asce)em.1943-7889.0000308) ~~
- ~~加速：MIMO(多输入多输出) MLP，可以加速训练。如果有多组实验的话，可以一次辨识出多组结构参数的区间，则可以加速推理。~~
- ~~Two-stage interval (of structural parameters) identification model. 先预测 X 的区间中心，后预测半径，两个都根据 Y 的区间来进行预测，虽然精度高，但是意义不大。~~
- 借鉴 GAN 思路，MLP1 进行 Interval identification，MLP2 进行 model validation(freezed surrogate model)。可以减小基于 MC 采样的不确定性传播带来的误差，导致经过IIM辨识的结构参数通过有限元计算出来的动力学响应不准确。 **MC 采样过程不可微**
  - 参考重参数化思路，将MC采样假设为均匀分布，从区间 $[a,b]$ 中采样 y 可以等效为：先从 $[0,1]$ 中随机抽取一个数 $x\sim U(0,1)$，然后根据 ($\frac{b-a}{1-0} =\frac{y-a}{x-0}$)，$y=(b-a)\times x+a$，这样梯度就可以反向传播到 a 和b上。
  - 通过对 $\hat{X}^{I}$ 和 $\hat{Y}^{I}$ 的两个约束来训练 IIM，可以使得 IIM 预测得到的结构参数经过有限元计算得到的动力学响应(Model Validation)更加接近真实的实验值。然而，由于不确定性传播的过程($X^{I} \to Y^{I}$)基于MC采样，数据集生成时得到的 $Y^{I}$ 与预测得到的 $\hat{Y^{I}}$ 之间必然会存在误差，在训练 IIM 的过程中，损失无法下降到最小。(只能添加一个限制: 对重参数化中的随机数$U(0,1)$控制相同的随机种子)

**不是创新的创新：Interval propagation + MLP + self-supervised learning** 

***self-supervised learning的说法有错误，真正的SSL应该不用先生成结构参数区间，再通过区间传播生成动力学响应区间。而是直接随机生成动力学响应区间***

真正的self-supervised learning是否有必要：
- 使用FE生成数据集虽然耗时，但是也是非常方便的，因此不用考虑小样本时如何训练的问题，因此自监督必要性不是很强

换个说法，不能叫自监督
1. Cyclic Feedback Supervision 但是这是使用了两个网络G和F，利用$X->G(\cdot)->Y->F(\cdot)->X$ 来监督X [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf) 
2. Proxy Feedback Supervision Network 代理模型通常用于替代复杂物理模型或昂贵的计算过程，在监督学习和强化学习中被广泛应用。代理模型的说法用在这有点混淆
3. Indirect Response Supervision 间接监督可以类比于**弱监督学习（Weakly Supervised Learning）**，其中监督信号不是直接来自目标变量，而是通过代理或间接变量进行 **也不太准确，因为不是间接变量，而是输入变量**
4. Physics Consistency Supervision Network [​内嵌物理知识神经网络（PINN）是个坑吗？ - 知乎](https://zhuanlan.zhihu.com/p/468748367?utm_psn=1787420267879907328) 有了这个说法了已经，PINNs是用网络通过约束自动求解PDE的 
  1. 论文1 [Integrating physics-informed machine learning with resonance effect for structural dynamic performance modeling - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352710224001955#sec1)
5. 物理反馈监督（Physics Feedback Supervised, PFS） Interval Calibration Model
6. **Dynamics Response-Guided** Interval Calibration Model
7. **Response-Consistent Supervised** Interval Calibration Model

就叫 Response-consistent Supervised Interval Model Calibration

主要思想：model updating/model calibration 的目的是让identified structural parameters x，经过有限元计算得到的dynamic response features y更加接近实验测量的$y_{mea}$。主要约束的是dynamic response features，而不是structural parameters。

围绕 fast interval identification：
1. ~~分层采样的方法，可以让采样点更加均匀，不确定性传播的结果更加准确(通过区间长度判断准确，如果经过MC-based UP，区间更长，则说明本次UP更加精确)~~
2. 将模型验证的过程作为另一个约束，添加到了训练过程中，使用重参数化解决了 MC 采样不可微/无法反向传播的问题。
3. 区间相似度损失函数
4. 在三个算例上对本文方法进行了验证

主要讨论的点：
- ~~Interval similarity loss 的权重$\lambda$~~
  - ~~当预测的区间与actual区间非常相近时，L1loss的平均~~
- 训练时loss value 的变化 以及 在测试集/实验集上测试结果的变化
- 修正的精度和时间(calibrated 结构参数 的结果和model validation 的结果)
- 有无Response consistent supervision 的影响
  - NICM：X的区间会快速收敛，但是可能预测的Y会有误差
  - RCSICM：Y的区间会快速收敛，但是可能X会有偏差

~~难点，在于做的东西是没有意义的：~~
- ~~不用Response consistent supervised也是可以的~~
- ~~不用interval similarity loss 更是可以的~~

---

- [ ] 检查流程框图的字符和文中的是否对应
- [ ] 缩进是否一致
- [ ] 表格图例 序号是否与文中对应
- [ ] 有过简写的，要用简写，检查简写问题，只出现一次

**几个容易被误解的地方**：
- 由于Interval propagation中使用了Reparameterization-based MC sampling 或者 interval perturbation method，但是如果使用FE模型，其照样是不可微分的，无法用于反向传播训练calibration model，因此必须要用FE surrogate model based on MLP/otherNN 来保证反向传播
- Interval Model Calibration是一种方法，model的含义应该是有限元模型，而Interval Calibration Model是基于MLP的网络模型，model的含义是网络模型。两者都同用一个单词model

术语简写：
Response-consistent supervised MLP-based Interval model calibration (MLP-IMC)
- MLP-IMC
- 我觉得还是RCSIMC好一点
Introduction后/relatedwork后，都用interval model calibration (第一章之后，全文都用interval model calibration)

确保缩写的首次出现
- finite element (FE)
- Monte Carlo (MC)
- Interval Response Surface Models (IRSM)
- Radial Basis Function (RBF)
- Back Propagation (BP)
- Interval Overlap Ratio (IOR)
- Interval Deviation Degree (IDD)
- Multi-Layer Perceptron (MLP)
- Relative Position Operator (RPO)
- response-consistent supervised Interval model calibration (RCS-IMC),
- interval calibration model (ICM)
- naive supervised interval model calibration (NS-IMC)




#### 数据集生成

```
k1 = np.random.uniform(k1_subint_lower, k1_subint_upper, N_sample_MC_subint)
k2 = np.random.uniform(k2_subint_lower, k2_subint_upper, N_sample_MC_subint)
k5 = np.random.uniform(k5_subint_lower, k5_subint_upper, N_sample_MC_subint)
ks = np.concatenate([np.expand_dims(k1,1), np.expand_dims(k2,1), np.expand_dims(k5,1)], axis=1)
```

```
MC   : [0.45585638 0.67835588 2.27067257 3.01441156 7.29330748 7.65326521 0.52790077 0.59090758]
MCsub: [0.45214765 0.68647469 2.23001174 3.03040203 7.29248957 7.66151722 0.55707975 0.56769328] ??? 最后一个响应特征的区间反而变小？
```

这种简单的子区间划分方法，会导致对参数值包含的情况的不完全(只考虑了：k1 小区间+k2 小区间+k3 小区间，k1 大区间+k2 大区间+k3 大区间。而没有考虑：k1 小区间+k2 大区间+k3 大区间等等其他不在对角线上的情况)
**这确实会导致子区间蒙特卡洛采样方法得到的响应区间不会变大，反而会变小**

经过修改代码，同时考虑不在多维变量对角线上的情况，MCsub+UP 得到的响应区间是在扩大的，这种方法虽然精度很高，但是缺点就是会发生维度爆炸，导致采样的时间过长。

```
MC   : [0.63652563 1.0503398  2.9200122  4.3315506  7.5164633  8.306494 0.5097574  0.6113596 ]
MCsub: [0.6295041  1.0680151  2.8079095  4.37455    7.5057774  8.309689 0.50374913 0.61328757]
```

与之相对的还有 LHS 方法，通过在低维上采样后，通过随机匹配的方法，可以节省时间，但同时也会降低精度。

```
# response features interval diameter = 2 * radius
MC    diameter: [0.41418052 1.4698663  0.7897434  0.10329747]
MCsub diameter: [0.43133098 1.5305772  0.7898903  0.10519242] # 精度高
LHS   diameter: [0.42642868 1.48715    0.78797436 0.10374653] # 速度快，精度可能会出现降低的情况
```

**可以讨论不同的 sub 数量和总采样数量的情况下，区间半径变大了多少**

#### MC samples

**两种方法得到的采样点是不同的**

```python
np.random.seed(1771)

# method1
a = np.random.uniform(0,1,10)
b = np.random.uniform(1,2,10)
c = np.random.uniform(2,3,10)
print(a,'\n',b,'\n',c)

[0.35119399 0.87023222 0.79901661 0.97782306 0.56443314 0.13076239 0.36250584 0.85412665 0.2742619  0.76300781]
[1.99999546 1.10825641 1.38642265 1.33386948 1.71665597 1.89852029 1.30053206 1.02680851 1.94387877 1.14519414]
[2.48435611 2.4497283  2.36097399 2.21444825 2.16375047 2.19614223 2.20986247 2.41881811 2.59317697 2.11514801]

# method2
random_samples = np.random.uniform(0,1,10)
lb_a = 0; ub_a = 1
lb_b = 1; ub_b = 2
lb_c = 2; ub_c = 3
a = lb_a + (ub_a - lb_a) * random_samples
b = lb_b + (ub_b - lb_b) * random_samples
c = lb_c + (ub_c - lb_c) * random_samples

[0.35119399 0.87023222 0.79901661 0.97782306 0.56443314 0.13076239 0.36250584 0.85412665 0.2742619  0.76300781]
[1.35119399 1.87023222 1.79901661 1.97782306 1.56443314 1.13076239 1.36250584 1.85412665 1.2742619  1.76300781]
[2.35119399 2.87023222 2.79901661 2.97782306 2.56443314 2.13076239 2.36250584 2.85412665 2.2742619  2.76300781]

print(a,'\n',b,'\n',c)
```


#### MIMO MLP

| Case       | SISO                                                                                                                                                                                                                                                                                                                                      | MIMO $\times 5$                                                                                                                                                                                                                                                                                                                                             |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MassSpring | $[\underline{\omega_{1}^{2}},\overline{\omega_{1}^{2}},\underline{\omega_{2}^{2}},\overline{\omega_{2}^{2}},\underline{\omega_{3}^{2}},\overline{\omega_{3}^{2}},\underline{\|\varphi(1,1)\|},\overline{\|\varphi(1,1)\|}] \to[\underline{k_{1}},\overline{k_{1}},\underline{k_{2}},\overline{k_{2}},\underline{k_{3}},\overline{k_{3}}]$ | $[\underline{\omega_{1}^{2}},\overline{\omega_{1}^{2}},\underline{\omega_{2}^{2}},\overline{\omega_{2}^{2}},\underline{\omega_{3}^{2}},\overline{\omega_{3}^{2}},\underline{\\varphi(1,1)\|},\overline{\|\varphi(1,1)\|}] \times 5 \to [\underline{k_{1}},\overline{k_{1}},\underline{k_{2}},\overline{k_{2}},\underline{k_{3}},\overline{k_{3}}] \times 5$ |
| SteelPlate | $[\underline{f_{1}},\overline{f_{1}},\underline{f_{2}},\overline{f_{2}},\underline{f_{3}},\overline{f_{3}},\underline{f_{4}},\overline{f_{4}},\underline{f_{5}},\overline{f_{5}}] \to [\underline{E},\overline{E},\underline{G},\overline{G}]$                                                                                            |                                                                                                                                                                                                                                                                                                                                                             |
| Airplane   | $[\underline{f_{1}},\overline{f_{1}},\underline{f_{2}},\overline{f_{2}},\underline{f_{3}},\overline{f_{3}},\underline{f_{4}},\overline{f_{4}},\underline{f_{5}},\overline{f_{5}}] \to [\underline{a},\overline{a},\underline{b},\overline{b},\underline{T},\overline{T}]$                                                                 |                                                                                                                                                                                                                                                                                                                                                             |

单纯将 SISO 的 MLP 变成 MIMO，虽然减少了训练时间，但是在某些位置的预测精度会下降。**这是由于数据集构建时，只是单纯将原来的 NXn，reshape 成了(N/5)X(nx5)**，这导致在某些位置的数据集包含的情况不完全。

*过于局限，只适用于 MLP+model frequency 这种情况*

#### TS-IIM

在 MassSpring 的实验来看，精度 1>3>2，网络结构 1 相当于根据 Y 的区间来预测 X 的区间。

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240724105539.png)

#### Enhanced IIM (self-supervised)

自监督的区间辨识方法思路：
由于模型修正的最终目的是让修正后的响应特征与测量的响应特征之间的差异最小化，因此可以设计一种自监督形式的区间辨识模型：
基于MLP的区间辨识模型根据仿真的响应特征区间预测出结构参数的区间后，通过可微分的不确定性传播，包括重参数化采样、基于MLP的代理模型(在训练区间辨识模型时，冻结代理模型的网络参数：基于MLP的代理模型为一个可微的过程，将代理模型的网络参数冻结后在训练区间辨识模型的过程中不会再改变其网络参数)和区间计算，得到预测的响应特征区间，通过计算预测的响应特征区间与仿真的响应特征区间之间的损失，然后反向传播到区间辨识模型的网络参数中，不断重复这个过程，最后得到训练好的区间辨识模型。

模型验证：训练后的区间辨识模型可以根据测量的响应特征区间快速计算得到校准后的结构参数区间，且校准后的结构参数区间经过不确定性传播(MC+有限元)得到的响应特征区间与测量的响应特征区间之间的误差会很小。

##### Question1

数据集和训练中的UP都是用相同的采样方式

在训练过程中，出现了以下现象：
- 只对structural parameters添加约束，虽然x预测的很好，但是预测drf时出现半径预测很好、中心点预测误差大的情况
- 只对dynamic response feature添加约束，虽然y预测的很好，但是预测sp时出现半径预测很好、中心点预测误差大的情况。

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240727133928.png)


**由于代码出现的错误**：
- 预测出来$\hat{X}^{I}$根据MC采样时，重参数化公式写错，写成了`x_samples = x_lower_bound + (x_lower_bound - x_upper_bound) * uniform_samples`, 实际上应该是`x_samples = x_lower_bound + (x_upper_bound - x_lower_bound) * uniform_samples`，这样会导致预测的x区间向大数移动，因为这样才保证了下边界减过uniform_samples后得到的x_samples，经过MCUP得到的$Y^{I}$与实际相同

## 公式

Interval:
$\mathbf{a}^{\mathrm{I}}$

Datasets:
$\mathcal{D} =\{\mathbf{X}^{\mathrm{I}}_i,\mathbf{Y}^{\mathrm{I}}_i\}_{i=1}^s$
$\{\mathbf{X}^{\mathrm{I}}_i,\mathbf{Y}^{\mathrm{I}}_i\}_{m=1}^M$ $\mathcal{D}$


FE
$\mathcal{F}_{\mathrm{\Theta}}$

$\mathcal{F}$
Network:
$\mathbf{\Theta}$

$\mathcal{L}_{l1loss}^{\mathrm{I}}$

$\mathcal{L}_{l1loss}^{\mathrm{c}}$


$IS=\frac{1}{1+\exp\{-RPO(A,B)\}}, RPO(A,B)=\frac{min\{\overline a ,\overline b\}-max\{\underline a,\underline b\}}{max\{L(A),L(B)\}}$

$SIS(A,B)|n_{sub}=\frac1{n_{sub}}\sum_{j=1}^{n_{sub}}\left\{1-ISF\left(A^{(j)},B^{(j)}\right)\right\}$
$SDI(A,B) = p(A\leqslant B)=\frac{\max(0,\text{len}(A)+\text{len}(B)-\max(0,a-\underline b))}{\text{len}(A)+\text{len}(B)}$


$\left.\mathrm{RPO}(A,B)=\begin{cases}\frac{(\overline{a}-\underline{b})}{\max\{len(A),len(B)\}},&\quad\text{Case 1,2}\\\\\frac{(\overline{a}-\underline{a})}{\max\{len(A),len(B)\}},&\quad\text{Case 3}\\\\\frac{(\overline{b}-\underline{b})}{\max\{len(A),len(B)\}},&\quad\text{Case 4}\\\\\frac{(\overline{b}-\underline{a})}{\max\{len(A),len(B)\}},&\quad\text{Case 5,6}\end{cases}\right.$


{$\mathcal{L}_{is}(\widehat{\mathbf{Y}}^{\mathbf Im},\mathbf{Y}^{\mathbf Im})=1-\frac{\min\{\overline{\widehat{\mathbf{Y}}^m},\overline{\mathbf{Y}^{m}}\}-\max\{\underline{\widehat{\mathbf{Y}}^{m}},\underline{\mathbf{Y}^{m}}\}}{\max\{len(\widehat{\mathbf{Y}}^{\mathbf Im}),len(\mathbf{Y}^{\mathbf Im})\}}$;}

$$\begin{align} 
\mathcal{F}\left( {{x_i^I}} \right) 
& = \mathcal{F}\left( {{x_i ^C}} \right) + {\left. {\sum\limits_{{p_1} = 1}^m {\frac{{\partial \mathcal{F}\left( x \right)}}{{\partial {x _{{p_1}}}}}} } \right|_{x _i^I = {x ^C},i \ne {p_1}}} \cdot \Delta {x _{{p_1}}} \\
& + \frac{1}{2}{\left. {\sum\limits_{{p_1} = 1}^m {\sum\limits_{{p_2} = 1}^m {\frac{{\partial {\mathcal{F}^2}\left( x \right)}}{{\partial {x _{{p_1}}}\partial {x _{{p_2}}}}}} } } \right|_{x _i^I = {x ^C},i \ne {p_1},{p_2}}} \cdot \Delta {x _{{p_1}}}\Delta {x _{{p_2}}} + \ldots \\
& + \frac{1}{n}{\left. {\sum\limits_{{p_1} = 1}^m \ldots \sum\limits_{{p_n} = 1}^m {\frac{{\partial {\mathcal{F}^n}\left( x \right)}}{{\partial {x _{{p_1}}} \ldots \partial {x _{{p_n}}}}}} } \right|_{x _i^I = {x ^C},i \ne {p_1}, \ldots ,{p_n}}} \cdot \Delta {x _{{p_1}}} \ldots \Delta {x _{{p_n}}} + R 
\end{align}$$

$\mathcal{F}(x_i^I)\quad \cong \widehat{\mathcal{F}}(x_i^I)=\mathcal{F}(x_i^C)+\sum_{p_1=1}^m\frac{\partial\mathcal{F}(x)}{\partial x_{p_1}}\Bigg|_{x_i^I=x^C,i\neq p_1}\cdot\Delta x_{p_1}$

$\mathcal{F}_i$

$\frac{\partial\mathcal{L}_{is}(A,B)}{\partial {\overline a}} = \frac{\partial{\frac{\overline a - \underline b}{\overline a - \underline a}}}{\partial \overline a} =$

$\begin{cases}\overline{\widehat{\boldsymbol{Y}}}=\mathcal{F}_{\Theta}(\boldsymbol{X}^C)+\sum_{p=1}^m\frac{\mathcal{F}_{\Theta}(x_p^c+\delta x_p)-\mathcal{F}_{\Theta}(x_p^c)}{\delta x_p}\Delta x_p\\\underline{\widehat{\boldsymbol{Y}}}=\mathcal{F}_{\Theta}(\boldsymbol{X}^C)-\sum_{p=1}^m\frac{\mathcal{F}_{\Theta}(x_p^c+\delta x_p)-\mathcal{F}_{\Theta}(x_p^c)}{\delta x_p}\Delta x_p&\end{cases}$

## References

UP(uncertainty propagation)
- Survey of Multifidelity Methods in Uncertainty Propagation, Inference, and Optimization
- Modern Monte Carlo Methods for Efficient Uncertainty Quantification and Propagation: A Survey
  - standard MC is a very time-consuming method, which makes its use **unfeasible for complex high-fidelity simulation**. Because it needs extensive model evaluations to obtain an accurate approximation
  - MC relies on the ability to sample from the assumed probability distribution easily, but doing so is not always possible.

[Keywords(Model Updating) Journal or book title(Computers & Structures) Content ID(271458) - Search | ScienceDirect.com](https://www.sciencedirect.com/search?qs=Model%20Updating&pub=Computers%20%26%20Structures&cid=271458)

1. [Finite element model updating of civil engineering infrastructures: A literature review | QUT ePrints](https://eprints.qut.edu.au/121681/) 土木有限元综述
2. [Dealing with uncertainty in model updating for damage assessment: A review - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327014004130) 有限元 损伤检测 综述
3. [Finite element model updating using deterministic optimisation: A global pattern search approach - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0141029618340665?via%3Dihub) 
4. [Finite Element Model Updating of Bridge Structures Based on Improved Response Surface Methods - Zhao - 2023 - Structural Control and Health Monitoring - Wiley Online Library](https://onlinelibrary.wiley.com/doi/10.1155/2023/2488951) 
5. [Stochastic Model Updating with Uncertainty Quantification: An Overview and Tutorial - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327023006921?via%3Dihub)  Airplane
6. [Interval model updating using universal grey mathematics and Gaussian process regression model - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327019306764?via%3Dihub)
7. [Stochastic model updating: Part 1—theory and simulated example - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327005000981)
8. [Stochastic model updating: Part 2—application to a set of physical structures - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327005001007)
9. [Numerical convergence and error analysis for the truncated iterative generalized stochastic perturbation-based finite element method - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0045782523001160)
10. [A response surface approach for the static analysis of stochastic structures with geometrical nonlinearities - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0045782503003797)
11. [Adaptive response surface based efficient Finite Element Model Updating - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0168874X13001789) Meta-Model method
12. [A two-step interval structural damage identification approach based on model updating and set-membership technique - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0263224121004516?via%3Dihub)
13. [An interval-based technique for FE model updating | International Journal of Reliability and Safety](https://www.inderscienceonline.com/doi/abs/10.1504/IJRS.2009.026836) 
14. [Interval model updating with irreducible uncertainty using the Kriging predictor](Interval%20model%20updating%20with%20irreducible%20uncertainty%20using%20the%20Kriging%20predictor.md) KP
15. [An interval model updating strategy using interval response surface models](An%20interval%20model%20updating%20strategy%20using%20interval%20response%20surface%20models.md) IRSM
16. [Interval model updating using perturbation method and Radial Basis Function neural networks](Interval%20model%20updating%20using%20perturbation%20method%20and%20Radial%20Basis%20Function%20neural%20networks.md) RBF 1
17. [Full article: A novel swarm intelligence optimization approach: sparrow search algorithm](https://www.tandfonline.com/doi/full/10.1080/21642583.2019.1708830) SSA 
18. [Model updating of an existing bridge with high-dimensional variables using modified particle swarm optimization and ambient excitation data - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S026322412030292X) PSO
19. [Finite element model updating using simulated annealing hybridized with unscented Kalman filter - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0045794916303935) SA
20. [Uncertainty Updating of Finite Element Models Using Interval Analysis | International Journal of Structural Stability and Dynamics](https://www.worldscientific.com/doi/abs/10.1142/S0219455420410126)
21. [Structural dynamics model updating with interval uncertainty based on response surface model and sensitivity analysis](https://www.tandfonline.com/doi/full/10.1080/17415977.2018.1554656#abstract)
22. [Modified perturbation method for eigenvalues of structure with interval parameters | Science China Physics, Mechanics & Astronomy](https://link.springer.com/article/10.1007/s11433-013-5328-6) perturbation propagation
23. [Interval parameter sensitivity analysis based on interval perturbation propagation and interval similarity operator - Archive ouverte HAL](https://hal.science/hal-04273667) perturbation propagation 1
24. [An improved interval model updating method via adaptive Kriging models | Applied Mathematics and Mechanics](https://link.springer.com/article/10.1007/s10483-024-3093-7) vertex method
25. [An Interval Model Updating Method Based on Meta-Model and Response Surface Reconstruction | International Journal of Structural Stability and Dynamics](https://www.worldscientific.com/doi/10.1142/S0219455423501158) vertex method 1
26. [A PCA-Based Approach for Structural Dynamics Model Updating with Interval Uncertainty | Acta Mechanica Solida Sinica](https://link.springer.com/article/10.1007/s10338-018-0064-0) MC 1
27. [MULTILEVEL QUASI-MONTE CARLO FOR INTERVAL ANALYSIS - International Journal for Uncertainty Quantification, 巻 12, 2022, 発行 4 - Begell House Digital Library](https://www.dl.begellhouse.com/jp/journals/52034eb04b657aea,324c50a10c1d9bd3,59014ac63d27ccc2.html)
28. [(PDF) Stochastic model updating based on sub-interval similarity and BP neural network](https://www.researchgate.net/publication/367239633_Stochastic_model_updating_based_on_sub-interval_similarity_and_BP_neural_network)  SSA BP 2
29. [Efficient Bayesian inference for finite element model updating with surrogate modeling techniques | Journal of Civil Structural Health Monitoring](https://link.springer.com/article/10.1007/s13349-024-00768-y)
30. [Interval identification of structural parameters using interval overlap ratio and Monte Carlo simulation - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S096599781731164X) IOR
31. [Interval Identification of Structural Parameters Using Interval Deviation Degree and Monte Carlo Simulation | International Journal of Computational Methods](https://www.worldscientific.com/doi/abs/10.1142/S0219876218501037) IDD
32. [A SATISFACTORY SOLUTION FOR INTERVAL NUMBER LINEAR PROGRAMMING - 百度学术](https://xueshu.baidu.com/usercenter/paper/show?paperid=753f8801c91a5e83d060717a30404554) L. Xinwang, A satisfactory solution for interval number linear programming, Journal of Systems Engineering, (1999). SDI
33. [The interval intersection method for FE model updating - IOPscience](https://iopscience.iop.org/article/10.1088/1742-6596/305/1/012091) S. Gabriele, The interval intersection method for FE model updating, Journal of Physics: Conference Series, 305 (2011) 012091. II
34. [A logical calculus of the ideas immanent in nervous activity | Bulletin of Mathematical Biology](https://link.springer.com/article/10.1007/BF02478259) 
35. [Learning representations by back-propagating errors | Nature](https://www.nature.com/articles/323533a0)
36. [A finite element-convolutional neural network model (FE-CNN) for stress field analysis around arbitrary inclusions - Astrophysics Data System](https://ui.adsabs.harvard.edu/abs/2023MLS%26T...4d5052R/abstract)
37. [Deep learning prediction of stress fields in additively manufactured metals with intricate defect networks - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167663621004026)
38. [Sequence-based modeling of deep learning with LSTM and GRU networks for structural damage detection of floating offshore wind turbine blades - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0960148121005371?via%3Dihub)
39. [Deep learning-based Structural Health Monitoring for damage detection on a large space antenna - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0094576521004082?via%3Dihub)
40. [Loss Functions and Metrics in Deep Learning | Abstract](https://arxiv.org/abs/2307.02694)
41. [The sub-interval similarity: A general uncertainty quantification metric for both stochastic and interval model updating - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327022004575?via%3Dihub) SIS
42. [Dynamic load identification based on deep convolution neural network - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327022008251#s0090)
43. [A feature map of frequency response functions based model updating method using the Bayesian convolutional neural network - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327023007264?ref=pdf_download&fr=RR-2&rr=87df480b9ca904d1#s0060)
44. [A Self-Supervised Representation Learner for Bearing Fault Diagnosis Based on Motor Current Signals | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/10623379)




Other paper about Self-supervised learning:
- [ConMLP: MLP-Based Self-Supervised Contrastive Learning for Skeleton Data Analysis and Action Recognition - PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10007586/)
- [A Self-Supervised Learning Based Channel Attention MLP-Mixer Network for Motor Imagery Decoding - PubMed](https://pubmed.ncbi.nlm.nih.gov/35976835/)
- [Self-supervised Feature Learning by Cross-modality and Cross-view Correspondences](https://arxiv.org/pdf/2004.05749)
- 


1. [Deterministic and probabilistic-based model updating of aging steel bridges - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352012423006239?via%3Dihub)
2. [Stochastic model updating: Part 1—theory and simulated example - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327005000981)
3. [Stochastic model updating: Part 2—application to a set of physical structures - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327005001007)
4. [Numerical convergence and error analysis for the truncated iterative generalized stochastic perturbation-based finite element method - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0045782523001160)
5. [A response surface approach for the static analysis of stochastic structures with geometrical nonlinearities - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0045782503003797)
6. [Adaptive response surface based efficient Finite Element Model Updating - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0168874X13001789)
7. 
8. [Interval model updating with irreducible uncertainty using the Kriging predictor](Interval%20model%20updating%20with%20irreducible%20uncertainty%20using%20the%20Kriging%20predictor.md)
9. [An interval model updating strategy using interval response surface models](An%20interval%20model%20updating%20strategy%20using%20interval%20response%20surface%20models.md) 
10. [Interval model updating using perturbation method and Radial Basis Function neural networks](Interval%20model%20updating%20using%20perturbation%20method%20and%20Radial%20Basis%20Function%20neural%20networks.md) RBF 1
11. [Full article: A novel swarm intelligence optimization approach: sparrow search algorithm](https://www.tandfonline.com/doi/full/10.1080/21642583.2019.1708830) SSA 
12. [Model updating of an existing bridge with high-dimensional variables using modified particle swarm optimization and ambient excitation data - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S026322412030292X) PSO
13. [Finite element model updating using simulated annealing hybridized with unscented Kalman filter - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0045794916303935) SA
14. [Modified perturbation method for eigenvalues of structure with interval parameters | Science China Physics, Mechanics & Astronomy](https://link.springer.com/article/10.1007/s11433-013-5328-6) perturbation propagation
15. [Interval parameter sensitivity analysis based on interval perturbation propagation and interval similarity operator - Archive ouverte HAL](https://hal.science/hal-04273667) **perturbation propagation 1**
16. [An improved interval model updating method via adaptive Kriging models | Applied Mathematics and Mechanics](https://link.springer.com/article/10.1007/s10483-024-3093-7) vertex method
17. [An Interval Model Updating Method Based on Meta-Model and Response Surface Reconstruction | International Journal of Structural Stability and Dynamics](https://www.worldscientific.com/doi/10.1142/S0219455423501158) **vertex method 1**
18. [A PCA-Based Approach for Structural Dynamics Model Updating with Interval Uncertainty | Acta Mechanica Solida Sinica](https://link.springer.com/article/10.1007/s10338-018-0064-0) **MC 1**
19. [MULTILEVEL QUASI-MONTE CARLO FOR INTERVAL ANALYSIS - International Journal for Uncertainty Quantification, 巻 12, 2022, 発行 4 - Begell House Digital Library](https://www.dl.begellhouse.com/jp/journals/52034eb04b657aea,324c50a10c1d9bd3,59014ac63d27ccc2.html)
20. [Interval identification of structural parameters using interval overlap ratio and Monte Carlo simulation - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S096599781731164X) IOR
21. [Interval Identification of Structural Parameters Using Interval Deviation Degree and Monte Carlo Simulation | International Journal of Computational Methods](https://www.worldscientific.com/doi/abs/10.1142/S0219876218501037) IDD
22. L. Xinwang, A satisfactory solution for interval number linear programming, Journal of Systems Engineering, (1999). SDI
23. S. Gabriele, The interval intersection method for FE model updating, Journal of Physics: Conference Series, 305 (2011) 012091. II
24. [(PDF) Stochastic model updating based on sub-interval similarity and BP neural network](https://www.researchgate.net/publication/367239633_Stochastic_model_updating_based_on_sub-interval_similarity_and_BP_neural_network)  SSA BP 2

[Interval model updating using perturbation method and Radial Basis Function neural networks](Interval%20model%20updating%20using%20perturbation%20method%20and%20Radial%20Basis%20Function%20neural%20networks.md) RBF 2

25. [Efficient Bayesian inference for finite element model updating with surrogate modeling techniques | Journal of Civil Structural Health Monitoring](https://link.springer.com/article/10.1007/s13349-024-00768-y)

[An improved interval model updating method via adaptive Kriging models | Applied Mathematics and Mechanics](https://link.springer.com/article/10.1007/s10483-024-3093-7) vertex method 2
[Interval parameter sensitivity analysis based on interval perturbation propagation and interval similarity operator - Archive ouverte HAL](https://hal.science/hal-04273667) perturbation propagation 2
[A PCA-Based Approach for Structural Dynamics Model Updating with Interval Uncertainty | Acta Mechanica Solida Sinica](https://link.springer.com/article/10.1007/s10338-018-0064-0) MC 2

续：(26~40)
1. MLP初始: [A logical calculus of the ideas immanent in nervous activity | Bulletin of Mathematical Biology](https://link.springer.com/article/10.1007/BF02478259) [Learning representations by back-propagating errors | Nature](https://www.nature.com/articles/323533a0)
2. 分类：[Sustainability | Free Full-Text | Machine Learning Approach Using MLP and SVM Algorithms for the Fault Prediction of a Centrifugal Pump in the Oil and Gas Industry](https://www.mdpi.com/2071-1050/12/11/4776), [A Data-Driven Fault Diagnosis Methodology in Three-Phase Inverters for PMSM Drive Systems | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/7565605), [Machines | Free Full-Text | Fault Detection and Diagnosis of the Electric Motor Drive and Battery System of Electric Vehicles](https://www.mdpi.com/2075-1702/11/7/713)
3. 回归：[A finite-element-informed neural network for parametric simulation in structural mechanics - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0168874X22001779) ，[Data-driven multiscale finite-element method using deep neural network combined with proper orthogonal decomposition | Engineering with Computers](https://link.springer.com/article/10.1007/s00366-023-01813-y), [Frequency Response Function‐Based Finite Element Model Updating Using Extreme Learning Machine Model - Zhao - 2020 - Shock and Vibration - Wiley Online Library](https://onlinelibrary.wiley.com/doi/10.1155/2020/8526933)
4. CNNs: [Dynamic load identification based on deep convolution neural network - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327022008251#s0090), [A feature map of frequency response functions based model updating method using the Bayesian convolutional neural network - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327023007264?ref=pdf_download&fr=RR-2&rr=87df480b9ca904d1#s0060), [A finite element-convolutional neural network model (FE-CNN) for stress field analysis around arbitrary inclusions - Astrophysics Data System](https://ui.adsabs.harvard.edu/abs/2023MLS%26T...4d5052R/abstract)，[Deep learning prediction of stress fields in additively manufactured metals with intricate defect networks - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167663621004026)
5. RNNs: [Sequence-based modeling of deep learning with LSTM and GRU networks for structural damage detection of floating offshore wind turbine blades - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0960148121005371?via%3Dihub)， [Deep learning-based Structural Health Monitoring for damage detection on a large space antenna - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0094576521004082?via%3Dihub)， [Merged LSTM-based pattern recognition of structural behavior of cable-supported bridges - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0952197623009582?via%3Dihub)

[Dynamic load identification based on deep convolution neural network - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327022008251#s0090) CNNs 1
[A feature map of frequency response functions based model updating method using the Bayesian convolutional neural network - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327023007264?ref=pdf_download&fr=RR-2&rr=87df480b9ca904d1#s0060) CNNs 2

续: (41~)
1. [The sub-interval similarity: A general uncertainty quantification metric for both stochastic and interval model updating - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327022004575?via%3Dihub) SIS
2. [An interval model updating strategy using interval response surface models - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327015000229?via%3Dihub) IRSM

