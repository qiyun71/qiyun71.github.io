---
title: Types of model updating
date: 2025-11-10 19:29:27
tags: 
categories: ModelUpdating
Year: 
Journal:
---

How many types about model updating with machine learning?

model updating: update parameters by minimizing the discrepancies between simulated and measured response.

<!-- more -->

# Stochastic model udpating

## Surrogate model

Frequency:
Optimization algorithm + distance，其中simulation需要代理模型
- [Bi et al., The role of the bhattacharyya distance in stochastic model updating, 2019-02, Mechanical Systems and Signal Processing](zotero://select/library/items/7JZEN6NP)
- [Zhao et al., Stochastic model updating based on sub-interval similarity and BP neural network, 2024-06-17, Mechanics of Advanced Materials and Structures](zotero://select/library/items/ZW54T7KV)

Bayesian likelihood:
Approximated Bayesian Computation，likelihood 是基于distance计算，其中simulation需要代理模型
- [Bi et al., Stochastic Model Updating with Uncertainty Quantification: An Overview and Tutorial, 12/2023, Mechanical Systems and Signal Processing](zotero://select/library/items/5JEKED2M)
- [Li et al., Bayesian finite element model updating with a variational autoencoder and polynomial chaos expansion, 2024-10-01, Engineering Structures](zotero://select/library/items/ZWX7K3HN)

likelihood计算方法创新：相当于使用VAE的编码器部分，将$\mathbf{x_{obs}}$降维到$\mathbf{z}$，代替$p(\mathbf{z}|\mathbf{x_{obs}})$部分的计算，然后通过将$\mathbf{x_{sim}}$同样降维到$\mathbf{z}$，代替$p(\mathbf{z}|\mathbf{x_{sim}})=p(\mathbf{z}|h(\theta))=p(\mathbf{z}|\theta)$的计算，然后两者共同计算likelihood用于Bayesian model updating
- [Lee et al., Latent space-based stochastic model updating, 2025-07, Mechanical Systems and Signal Processing](zotero://select/library/items/48BE68BJ)

## Inverse surrogate model

From response to parameters
- [Bi et al., Stochastic model calibration with image encoding: Converting high-dimensional sequential responses into RGB images for neural network inversion, 05/2025, Mechanical Systems and Signal Processing](zotero://select/library/items/DZDNR6RY)
- [Wang et al., A feature map of frequency response functions based model updating method using the Bayesian convolutional neural network, 2023-12, Mechanical Systems and Signal Processing](zotero://select/library/items/ZTZZWGHM)
- [Zhang et al., A frequency response model updating method based on unidirectional convolutional neural network, 2021-07-18, Mechanics of Advanced Materials and Structures](zotero://select/library/items/5Q59AHMI)

## Generative model

Flow-based：
- [Wang et al., Data-driven stochastic model updating and damage detection with deep generative model, 2025-06, Mechanical Systems and Signal Processing](zotero://select/library/items/BQV3XQBN)
- [Zeng et al., A recursive inference method based on invertible neural network for multi-level model updating using video monitoring data, 2023-11, Mechanical Systems and Signal Processing](zotero://select/library/items/J2QL4NAV)

GAN：生成器不断生成样本，判别器来判断该样本的响应是否与输出一致
- [Mo et al., Enhancing high-dimensional probabilistic model updating: A generic generative model-inspired framework with GAN-embedded implementation, 2025-10-01, Computer Methods in Applied Mechanics and Engineering](zotero://select/library/items/2C5LQATL)


VAE：先训练代理模型作为Decoder，然后训练VAE的Encoder和Bottleneck，输出updated state/parameters
- [Zhao et al., Reverse Variational Autoencoder: A probabilistic inference framework for structural health monitoring and inverse analysis in engineering, 2025-11-09, Reliability Engineering & System Safety](zotero://select/library/items/Z9UYP25N)
Self-supervised：先训练代理模型，然后使用响应一致性监督训练逆代理模型
- 待发
测试时训练方法：先训练AE，然后冻结Bottleneck和Decoder部分，将Encoder替换为更复杂的网络在线训练
- [Li et al., Fault detection in nonstationary industrial processes via kolmogorov-arnold networks with test-time training, 2025-10-24, ISA Transactions](zotero://select/library/items/2JTFKDQQ) 原文中是一个异常点探测的任务，先用历史数据训练AE，然后用Encoder和Bottleneck部分得到大致的latent 分布并取合适阈值作为异常判断。训练时冻结Bottleneck和Decoder部分，只训练Encoder部分，以适应数据分布的变化。

## Reforcement learning

RL：强化学习的思路来进行模型修正
- [Wang et al., RT-FEMU: A reinforcement and transfer learning-based framework for continuous finite element model updating, 2025-10-27, Journal of Building Engineering](zotero://select/library/items/5T36MBG7)