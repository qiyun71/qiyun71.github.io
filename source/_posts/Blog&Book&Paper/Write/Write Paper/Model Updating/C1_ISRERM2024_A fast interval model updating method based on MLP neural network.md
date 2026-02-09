---
title: C1_ISRERM2024_A fast interval model updating method based on MLP neural network
date: 2025-12-26 14:17:54
tags: 
categories: Blog&Book&Paper/Write/Write Paper/Model Updating
Year: 
Journal:
---
| 区别        | 会议                                                                | SCI                                                                                                    |
| --------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| MLP-based | 动力学响应-->结构参数，单次对单次                                                | 动力学响应-->结构参数，区间对区间                                                                                     |
| MLP       | Inverse surrogate model                                           | Interval identification model                                                                          |
| 算例        | 三自由度弹簧、钢板                                                         | 三自由度弹簧、钢板、飞机                                                                                           |
| 期刊        | ISRERM                                                            | [Chinese Journal of Aeronautics](https://www.sciencedirect.com/journal/chinese-journal-of-aeronautics) |
|           | A fast interval model updating method based on MLP neural network | Interval Model Calibration with Response-Consistent Supervised Learning Network                        |

<!-- more -->

# Conference

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

