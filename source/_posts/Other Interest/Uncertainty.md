---
title: Uncertainty
date: 2023-12-15 21:31:34
tags:
  - Uncertainty
categories: Other Interest
---

Uncertainty in everything

<!-- more -->

# Uncertainty

> [浅谈Epistemic Uncertainty 和 Aleatoric Uncertainty - 知乎](https://zhuanlan.zhihu.com/p/656915794)
> [【实验笔记】深度学习中的两种不确定性（上） - 知乎](https://zhuanlan.zhihu.com/p/56986840)

不确定性根据系统内的不确定性来源可以分为Aleatoric Uncertainty和Epistemic Uncertainty，Aleatoric Uncertainty通常指的是数据不确定性，往往来自于数据本身的randomness或variability，是数据固有的一种属性。Epistemic uncertainty通常指的是模型不确定性或认知不确定性，其不确定性通常来源于缺乏足量信息的支撑。

> [Stochastic Model Updating with Uncertainty Quantification: An Overview and Tutorial - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327023006921?via%3Dihub)

Uncertainty sources：Parameter uncertainty、Model form uncertainty、Experiment uncertainty
根据参数是否存在认知epistemic和/或选择性(偶然)aleatory不确定性，**将不确定性参数分为四类**：
- 既不具有认知不确定性，也不具有选择性不确定性的参数被表示为具有完全确定值的常数。
- 只有认知不确定性的参数被表示为一个未知但固定的常数，落在预定义的区间内。
- 将仅具有aleatory偶然不确定性的参数表示为具有完全确定的分布性质(如分布格式、均值、方差等)的随机变量。这种完全确定的分布称为“精确概率”。
- 同时具有认知不确定性和选择性不确定性的参数被表示为一个分布性质不完全确定的随机变量，即“不精确概率”。这种不精确的概率由所谓的概率盒(P-box)来建模，其中无限数量的累积分布函数(CDF)曲线构成概率空间中的特定区域。

>  [A Survey of Uncertainty in Deep Neural Networks](https://arxiv.org/pdf/2107.03342)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241215213725.png)


# Seminar

## 现代机器学习视角下的不确定性度量

> [\[FAI\] 清华 滕佳烨 | 现代机器学习视角下的不确定性度量 | ICLR 23_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1wu411H7uS/?vd_source=1dba7493016a36a32b27a14ed2891088)
> [FAI-Seminar](https://www.fai-seminar.ac.cn/)

之前的不确定性：Parameter Inference
神经网络的：Predictive Inference

Parameter Inference：(线性回归任务)
- $y=x^{\top}\beta^{*}+\epsilon,\epsilon{\sim}\mathcal{N}\left(0,\sigma^{2}I\right).$
- $\hat{\beta}=\left(X^{\top}X\right)^{-1}X^{\top}Y\sim\mathcal{N}\left(\beta^{*},\sigma^{2}\left(X^{\top}X\right)^{-1}\right).$
- Confidence Band $\mathcal{C}_{1-\alpha}(X)=\hat{\beta}\pm c_{\alpha}\sigma\left(X^{\top}X\right)^{-1/2},$
  - 置信度$\alpha$，对应常数$c_{\alpha}$，则$\mathbb{P}\left(\beta^{*}\in\mathcal{C}_{1-\alpha}(X)\right)\geq1-\alpha.$

Predictive Inference：
由于神经网络的参数都是很难解释的，无法Parameter Inference，或者说无法通过简单的Parameter Inference得到:
$y=x^{\top}\beta^{*}+\epsilon,\epsilon{\sim}\mathcal{N}\left(0,\sigma^{2}I\right).$能构造Parameter cb：$\mathcal{C}_{1-\alpha}(X)=\hat{\beta}\pm c_{\alpha}\sigma\left(X^{\top}X\right)^{-1/2},\beta^{*}\in\mathcal{C}_{1-\alpha}(X)\mathrm{~whp}.$，然后根据$\beta^{*}$的band构造$y$的band
- 需要构造两次band
- parameter通常是是无意义的，其band也没意义

Predictive Inference定义：
给定一个在training set $\mathcal{D}=\{(X_{i},Y_{i})\}$上训练好的model $\mu$，以及一个置信度$\alpha$，构建一个置信band $\mathcal{C}_{1-\alpha}(\cdot;\mathcal{D},\mu)$，对任意测试point $(X^\prime,Y^\prime)$有：
$\mathbb{P}\left(Y^{\prime}\in\mathcal{C}_{1-\alpha}(X^{\prime};\mathcal{D},\mu)\right)\geq1-\alpha.$

Generalization(Validation Trick: 将验证集的loss当作测试集的loss)
$\mathbb{E}\|\widehat{Y}-Y\|\leq\epsilon$ 越小越好
Predictive Inference
$\mathbb{P}\left(Y^{\prime}\in\mathcal{C}_{1-\alpha}(X^{\prime};\mathcal{D},\mu)\right)\geq1-\alpha.$ 满足大于$1-\alpha$的band越小越好

Generalization(Validation Trick） --> Predictive Inference(Conformal Prediction):
- Training set上训练model
- Calibration set上测试model
