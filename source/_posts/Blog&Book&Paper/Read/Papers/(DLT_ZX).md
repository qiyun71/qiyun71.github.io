---
title: (DLT_ZX)
date: 2024-12-07 20:27:14
tags:
  - 
categories: Blog&Book&Paper/Read/Papers
---

[北京航空航天大学主页平台系统 董雷霆--中文主页--首页](https://shi.buaa.edu.cn/dongleiting/zh_CN/index/16082/list/index.htm)
[Xuan Zhou](https://xuanzhou.ac.cn/)

<!-- more -->

[Copula-based Collaborative Multi-Structure Damage Diagnosis and Prognosis for Fleet Maintenance Digital Twins](https://xuanzhou.ac.cn/zh/publication/aiaaj2023a/aiaaj2023a.pdf)

>  [粒子滤波器 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E7%B2%92%E5%AD%90%E6%BF%BE%E6%B3%A2%E5%99%A8)

粒子滤波(particle filter, PF)可以建模非高斯非线性过程(包含认知和随机不确定性)，被广泛用于airframe 数字孪生中
**研究意义**：目前，PF-based digital twin主要集中在单个结构的诊断和预测，很少关注包含多个结构的舰队水平(许多情况下，损伤状态与不同结构有关)。It is desirable to develop an approach to efficiently
consider the correlation between structures within the fleet and improve the holistic fleet diagnosis and prognosis.
**研究内容**：novel copula-based approach
- utilizing the copulas to model the dependence between crack length distributions to obtain an approximate joint probability distribution for collaborative updating
- 


[A fuzzy-set-based joint distribution adaptation method for regression and its application to online damage quantification for structural digital twin - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327023000717?via%3Dihub)

![featured_hudc01f57e244ced01ea723c3af8d576db_1478949_720x2500_fit_q75_h2_lanczos_3.webp (720×540)|555](https://xuanzhou.ac.cn/publication/mssp2023/featured_hudc01f57e244ced01ea723c3af8d576db_1478949_720x2500_fit_q75_h2_lanczos_3.webp)

在线损伤量化，insufficient labeled data --> 利用从类似结构/损伤的历史标签数据 or 仿真的数字孪生数据， 对现有的诊断任务有益。 然而，大多方法适用于分类任务，而不适用于回归任务。本文利用fuzzy set将离散的分类标签转换成连续的实值标签，提出新的 domain adaptation (eg迁移学习)方法，提出Online Fuzzy-setbased Joint Distribution Adaptation方法来进行回归任务 在线更新数据分布的模糊集(变化的)

数字孪生DT-->SHM-->通过传感器数据的Online damage quantification
In fleet，由于制造和使用过程中的差异，每个组件经历的损伤演化过程各不相同 --> 期望对每个组件单独进行损伤量化。然而在线阶段工程中enough labeled data是很难获取的 --> 可以从先前的实验或仿真中获取的数值模型和数据来进行当前的损伤量化。However, because of the problem-dependent nature of the data and models, adapting themto new structures or damage is challenging.
--> **Domain adaptation methods**： 
- Transfer Component Analysis (TCA） evaluates the marginal distribution difference in two domain
- Joint Distribution Adaptation (JDA) considers the marginal and conditional distribution discrepancies simultaneously and outperforms TCA in most circumstances.

然而现有的方法(In DT / SHM)主要集中在损伤探测，是一个分类问题:
- http://dx.doi.org/10.1117/12.882025. transfer learning
- http://dx.doi.org/10.3929/ethz-b-000506104. Feature Alignment Neural Network 
- http://dx.doi.org/10.1016/j.ymssp.2021.108426. integrated the adaptation-regularization-based transfer learning (ARTL) into the Bayesian model updating (BMU)
- http://dx.doi.org/10.1177/14759217221094500. TCA to improve the accuracy of the damage detection
- http://dx.doi.org/10.1016/j.ymssp.2020.107141. http://dx.doi.org/10.1016/j.ymssp.2020.107144. http://dx.doi.org/10.1016/j.ymssp.2020.107142. application of domain adaptation in SHM (vibration-based damage detection) ： opulation-Based Structural Health Monitoring (PBSHM).
- http://dx.doi.org/10.1016/j.ymssp.2022.108991. http://dx.doi.org/10.1007/s10489-022-03713-y. ： deep learning
- http://dx.doi.org/10.1016/j.jsv.2021.116072. http://dx.doi.org/10.1007/s13349-022-00565-5. Unsupervised damage detection

--> Damage Quantification
- http://dx.doi.org/10.1177/14759217221094500. TCA and Gaussian process regression (GPR),  but the conditional distribution discrepancy is not considered.
- http://dx.doi.org/10.3390/s18061881. http://dx.doi.org/10.1016/j.conbuildmat.2020.119096. vision-based

然而instance的样本数量少依然是一个关键问题，此外：
- the response (features) of damage sizes (labels) in different locations or structures has significant variations, indicating different conditional distributions
- the label is continuous real-value instead of discrete one-hot, methods that also considering conditional distribution discrepanc

为此：
-  http://dx.doi.org/10.1109/TFUZZ.2016.2633379. 使用online weighted adaptation regularization实现脑电图信号的driver drowsiness estimation。**fuzzy sets** (FSs) are used to transform the continuous label into several fuzzy classes。
- http://dx.doi.org/10.1016/j.knosys.2021.107216. conditional distribution deep adaptation regression (CDAR)

--> the Online Fuzzy-set-based Joint Distribution Adaptation for Regression (OFJDAR)
- the regression problem is converted to a classification problem using **fuzzy sets**
- an **online damage quantification framework** for the structural component digital twin with domain adaptation is proposed
- Three types of domain adaptation in the damage quantification, including different locations of damage, different types of damage, and simulated vs. true damage, are carried out, and the results are compared with several baselines and discussed in detail.

Online Fuzzy-set-based Joint Distribution Adaptation for Regression: (Math)
- Problem definition
- Marginal distribution adaptation
- Conditional distribution adaptation by converting the regression task to a classification task
  - Computing the membership degree of samples with Fuzzy sets
  - Constructing the MMD matrices $\tilde{\mathbf{M}}_{(c)}$ incorporating fuzzy membership degree
- Finding the transformation by solving a constraint optimization problem
- OFJDAR algorithm
- Optimization of hypermeter 𝛾

