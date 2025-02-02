---
title: "Stochastic Model Calibration with Image Encoding: Converting High-Dimensional Frequency-Domain Responses into RGB Images for Neural Network Inversion"
date: 2024-05-24 09:33:02
tags: 
categories:
---

| Title     | Stochastic Model Calibration with Image Encoding: Converting High-Dimensional Frequency-Domain Responses into RGB Images for Neural Network Inversion |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    |                                                                                                                                                       |
| Conf/Jour |                                                                                                                                                       |
| Year      |                                                                                                                                                       |
| Project   |                                                                                                                                                       |
| Paper     |                                                                                                                                                       |

<!-- more -->

# 大纲

Introduction
- The Verification and Validation (V&V) process 努力缩小 Sim 和 Exp 之间的差异 --> 由于各种偏差，引入了不确定性
- 随机模型修正来辨识受 Aleatory Uncertainty 影响的参数，但是所用的 MC 方法会有计算负担
- 随机模型修正主要分为基于优化和基于贝叶斯的方法 --> 当新的试验数据可用时，修正过程会出现明显延迟，阻碍信息的及时提取
- 代理模型的使用 --> 然而其也有缺陷：过拟合、参数缩放困难、尤其在高维场景(例如 FRFs 和时序数据)
- 传统修正方法无法用到实时领域 --> 并且：容易陷入局部最优、需要手工设置超参数(正则化项) 修正好结果是非常耗时的
- 神经网络技术的发展，可以很方便地将高维数据处理成低维数据
- 最近地一些工作通过神经网络，输入 FRFs 来直接获得 updated structural parameters --> 然而目前方法主要集中在确定性模型修正中 
- 在处理 FRFs 和时序数据中，CNN 和 RNN 方法被引入 --> However, when it comes to handling high-dimensional frequency-domain or time-domain data, these methods face significant limitations. Specifically, neural networks often struggle with the sheer complexity of such data, leading to challenges in accuracy and computational efficiency. 😵不太对

Model calibration problem with frequency-domain quantity of interest
- 2.1 Deterministic model calibration
- 2.2 Stochastic model calibration 

Image conversion of structure frequency response data in neural network framework
- 3.1 VGG framework for model frequency response output
- 3.2 LSTM-ResNet framework design for model temporal sequences output

Data image storage operator for multi-node time- or frequency-domain data

Stochastic model calibration with inverse neural network

Case study: The NASA challenge problem 时序数据
- 6.1 Training data generation 
- 6.2 Neural network framework design for temporal sequences output
- 6.3 Model calibration based on LSTM-ResNet

Case study: The satellite FE Model 
- 7.1 Training data generation and its data image storage
- 7.2 Neural network framework design for FRF
- 7.3 Model calibration based on VGG11

Conclusion and perspectives

# Basic Information

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

