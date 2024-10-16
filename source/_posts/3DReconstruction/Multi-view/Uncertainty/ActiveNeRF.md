---
title: ActiveNeRF
date: 2024-10-10 14:22:31
tags:
  - 
categories: 3DReconstruction/Multi-view/Uncertainty
---

| Title     | ActiveNeRF                                                                                                      |
| --------- | --------------------------------------------------------------------------------------------------------------- |
| Author    | Pan, Xuran and Lai, Zihang and Song, Shiji and Huang, Gao                                                       |
| Conf/Jour | ECCV                                                                                                            |
| Year      | 2022                                                                                                            |
| Project   | [LeapLabTHU/ActiveNeRF: Official repository of ActiveNeRF (ECCV2022)](https://github.com/LeapLabTHU/ActiveNeRF) |
| Paper     | [ActiveNeRF: Learning where to See with Uncertainty Estimation](https://arxiv.org/pdf/2209.08546v1)             |

<!-- more -->


# AIR

If with incomplete scene observation, the original NeRF framework tends to collapse to trivial solutions by predicting the volume density as 0 for the unobserved regions. MLP在训练集中没包含的区域预测的 volume density 倾向于为0

***这种方法网络不会将高斯分布的mean值预测为GT color，然后将variance 预测为0吗?***

# Method

本文将emitted radiance value of each location in the scene建模为一个高斯分布，预测的variance可以看作aleatoric uncertainty的反映. Through this, the model is imposed to provide larger variances
in the unobserved region instead of collapsing to the trivial solution

在 $r(t)$位置的color被定义为高斯分布：mean=$\bar{c}(r(t))$, variance=$\bar{\beta}^{2}(r(t))$，并分别用两个网络来进行预测

$[\sigma,f,\beta^{2}(\mathrm{r}(t))]=\mathrm{MLP}_{\theta_{1},\theta_{3}}(\gamma_{\mathrm{x}}(\mathrm{r}(t)))$
$\bar{c}(\mathrm{r}(t))=\mathrm{MLP}_{\theta_{2}}(f,\gamma_{\mathrm{d}}(\mathrm{d}))$

如果 each point color 服从高斯分布，则rendered pixel color 也服从高斯分布： 
$\begin{aligned}\hat{C}(\mathrm{r})&\sim\mathcal{N}(\sum_{i=1}^{N_s}\alpha_i\bar{c}(\mathrm{r}(t_i)),\sum_{i=1}^{N_s}\alpha_i^2\bar{\beta}^2(\mathrm{r}(t_i)))\sim\mathcal{N}(\bar{C}(\mathrm{r}),\bar{\beta}^2(\mathrm{r})),\end{aligned}$

## Loss Function

优化的目标是让 “pixel color mean 与 GT之间的差异” 与 “pixel color variance” 最小：

假设两条光线之间相交的可能性很小，则rendered rays可以被假设为是独立的，因此the negative log likelihood最小：
$\min\limits_{\theta}-\log p_{\theta}(\mathcal{B}) =-\frac{1}{N}\sum\limits_{i=1}^{N}\log p_{\theta}(C(r_{i})) = \frac{1}{N}\sum\limits_{i=1}^{N}\frac{\mid\mid C(r_{i})-\bar{C}(r_{i})\mid\mid_{2}^{2}}{2\bar{\beta}^{2}(r_{i})}+\frac{\log\bar\beta^{2}(r_{i})}{2}$
- 寻找到最优的网络参数$\theta$，使得在这个$\theta$下估计的color的似然最大，即negative log likelihood最小
- 但是单纯使用negative log likelihood作为损失函数，会出现：网络预测的一条光线上的点的weight $\alpha_{i}$相近，**导致在物体表面出现blur**，加入正则化项，让每条光线上的平均密度$\sigma(r(t))$最小

加入正则化项后的loss function：
$\mathcal{L}^{uct}=\frac1N\sum_{i=1}^N\left(\frac{\|C(\mathrm{r}_i)-\bar{C}(\mathrm{r}_i)\|_2^2}{2\bar{\beta}^2(\mathrm{r}_i)}+\frac{\log\bar{\beta}^2(\mathrm{r}_i)}{2}+\frac\lambda{N_s}\sum_{j=1}^{N_s}\sigma_i(\mathrm{r}_i(t_j))\right),$

此外最终的loss还要添加 color 的 MAE 损失：
$\mathcal{L}^{uct}(C(\mathrm{r}),\bar{C}^f(\mathrm{r}))+\frac1N\sum_{i=1}^N\|C(\mathrm{r}_i)-\hat{C}^c(\mathrm{r}_i)\|_2^2.$

