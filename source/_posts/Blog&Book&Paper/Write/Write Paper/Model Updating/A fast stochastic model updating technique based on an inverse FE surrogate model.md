---
title: A fast stochastic model updating technique based on an inverse FE surrogate model
date: 2024-05-24 09:33:02
tags: 
categories:
---

| Title       | A fast stochastic model updating technique based on an inverse FE surrogate model |
| ----------- | ------------------- |
| Author      |                     |
| Conf/Jour   |                     |
| Year        |                     |
| Project     |                     |
| Paper       |                     |

<!-- more -->

## A fast stochastic model updating technique based on an inverse FE surrogate model

**随机模型修正** [Call for papers - Engineering Structures | ScienceDirect.com by Elsevier](https://www.sciencedirect.com/journal/engineering-structures/about/call-for-papers#computational-methods-for-stochastic-engineering-dynamics)
- **提交截止日期为 2024 年 4 月 15 日**
- 录取截止日期为 2024 年 6 月 15 日。

方法：CNN+RNN
算例：NASA 挑战、卫星

讨论分析：
- NASA 修正精度高，与其他方法进行对比
- 卫星可以实时修正，神经网络很适合做黑箱，训练时间长

确定模型修正：$θ={θ_i,i=1,2,…,N_θ}$ and $y={y_j,j=1,…,N_y }$
- 前向过程：$y=\mathbf{F}_M(\theta)$
- 优化目标：$\widehat{\theta}=\arg\min\mathbf{G}\left(\mathbf{F}_M(\theta),\mathbf{\varepsilon}_M(\mathbf{y}_{sim},\mathbf{y}_{exp})\right)$
随机模型修正：$\theta^R=\left\{\theta_i^R,i=1,2,...,N_\theta\right\},$ 
$\boldsymbol{y}_{sim}^{\boldsymbol{R}}=\left\{\boldsymbol{y}_{j}^{R},j=1,\ldots,N_{y}\right\}_{sim}$ and $\mathbf{y}_{j}^{R}=\left\{y_{1},y_{2},\ldots,y_{n_{sim}}\right\}^{T}$
$y_{exp}^{R}=\left\{y_{k}^{R},k=1,\ldots,N_{y}\right\}_{exp}$ and $y_k^R=\{y_1,y_2,…,y_{n_{exp}} \}^T$
- 前向过程：
- 优化目标：$\left.\widehat{\theta^R}\in\theta^R=\arg\min\mathbf{G}\left(\mathbf{F}_M(x,\theta^R),d(y_{sim}^R,y_{exp}^R)\right.\right)$

过程：
- sensitivity analysis 得到 the most critical parameters $θ={θ_i,i=1,2,…,N_θ}$
- 生成训练数据：每一行输出 y 对应每一行的输入 $\theta$
$y_{sim}(\zeta)=\{y_{sim}^1,y_{sim}^2,...,y_{sim}^{Nmc}\}^T\to\theta=\{\theta^1,\theta^2,...,\theta^{Nmc}\}^T$

$\boldsymbol{y}_{sim}(\zeta)=\begin{bmatrix}y_1^1(\zeta)&...&y_j^1(\zeta)&...&y_{Ny}^1(\zeta)\\y_1^2(\zeta)&...&y_j^2(\zeta)&...&y_{Ny}^2(\zeta)\\...&...&...&...&...\\y_1^{Nmc}(\zeta)&...&y_j^{Nmc}(\zeta)&...&y_{Ny}^{Nmc}(\zeta)\end{bmatrix}•\boldsymbol{\theta}=\begin{bmatrix}\theta_1^1&...&\theta_l^1&...&\theta_{N_\theta}^1\\\theta_1^2&...&\theta_l^2&...&\theta_{N_\theta}^2\\...&...&...&...&...&...\\\theta_1^{Nmc}&...&\theta_i^{Nmc}&...&\theta_{N_\theta}^{Nmc}\end{bmatrix}$

