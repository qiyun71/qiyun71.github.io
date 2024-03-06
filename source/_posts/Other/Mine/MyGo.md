
# A fast stochastic model updating technique based on an inverse FE surrogate model

**随机模型修正** [Call for papers - Engineering Structures | ScienceDirect.com by Elsevier](https://www.sciencedirect.com/journal/engineering-structures/about/call-for-papers#computational-methods-for-stochastic-engineering-dynamics)
- **提交截止日期为2024年4月15日**
- 录取截止日期为2024年6月15日。

方法：CNN+RNN
算例：NASA挑战、卫星

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
- sensitivity analysis得到the most critical parameters$θ={θ_i,i=1,2,…,N_θ}$
- 生成训练数据：每一行输出y对应每一行的输入$\theta$
$y_{sim}(\zeta)=\{y_{sim}^1,y_{sim}^2,...,y_{sim}^{Nmc}\}^T\to\theta=\{\theta^1,\theta^2,...,\theta^{Nmc}\}^T$

$\boldsymbol{y}_{sim}(\zeta)=\begin{bmatrix}y_1^1(\zeta)&...&y_j^1(\zeta)&...&y_{Ny}^1(\zeta)\\y_1^2(\zeta)&...&y_j^2(\zeta)&...&y_{Ny}^2(\zeta)\\...&...&...&...&...\\y_1^{Nmc}(\zeta)&...&y_j^{Nmc}(\zeta)&...&y_{Ny}^{Nmc}(\zeta)\end{bmatrix}•\boldsymbol{\theta}=\begin{bmatrix}\theta_1^1&...&\theta_l^1&...&\theta_{N_\theta}^1\\\theta_1^2&...&\theta_l^2&...&\theta_{N_\theta}^2\\...&...&...&...&...&...\\\theta_1^{Nmc}&...&\theta_i^{Nmc}&...&\theta_{N_\theta}^{Nmc}\end{bmatrix}$


# ISRERM EI 论文

 **区间模型修正**
 
方法：神经网络
算例：三自由度弹簧、钢板
- 查找一些区间模型修正的算例，sci国兆谱

Title: A interval model updating method based on Deep Neural Network

## Abstract

~~近年来，区间模型修正在实际工程中得到了广泛应用，尤其是在对结构系统认知不足的情况下。传统的区间模型修正方法存在优化时间长、输出高维特征时很难找到有效的不确定性量化指标等问题。DNN理论上能拟合任意的函数，基于此，本文提出了一种基于DNN的快速区间模型修正框架，将传统方法的逆问题转换为正问题求解。该框架将构建了以模型特征为输入来预测相应结构参数的网络结构，只需要简单的量化指标就可以训练网络。此外，训练完成后DNN推理的速度非常快，可以用于实时模型修正任务。最后，通过对质量弹簧系统和钢板结构这两个经典数值算例的验证，证明了所提出方法的可行性和有效性。~~

m202210465@xs.ustb.edu.cn
zhaoyanlin@ustb.edu.cn
ydb@ustb.edu.cn

Interval model updating is widely used for the case of insufficient knowledge of the structural system. Traditionalinterval model updating methods rely on optimization algorithms to updated the interval bounds of uncertain parameters. However, this approach has limitations like long optimization times and the inability to identify suitable uncertainty quantifiers for high-dimensional output features. 
~~MLP can theoretically fit arbitrary functions.~~
Hence, an fast interval model updating framework based on MLP (Multi-Layer Perceptron) neural network is proposed, which takes known model features as inputs to predict the corresponding structural parameters. 
本框架构建了基于MLP反向代理模型，将传统方法的反问题转换为正问题来求解。根据大量的模型特征和结构参数数据对，通过前向计算和反向传播，最终拟合出一个准确的反向代理模型。
The network training only requires the construction of simple metrics, and the network correction is very fast after the training is completed. 


## 会议流程

International Symposium on Reliability Engineering and Risk Management(ISRERM)
- October 18-21, 2024, Hefei, China
- 注册费、会议流程

| 时间                          | 事件                                                                          |
| --------------------------- | --------------------------------------------------------------------------- |
| Jan. 21 - **Feb. 29**, 2024 | **Submission of abstracts** for review                                      |
| Mar. 1 - Mar. 20, 2024      | Review of all abstracts                                                     |
| Mar. 21 - **May. 15**, 2024 | **Submission of full papers**                                               |
| May. 1 - **Sept. 1**, 2024  | **Registration with payment** required to be scheduled for the presentation |
| May. 1 - Oct.21, 2024       | Registration for ISRERM 2024                                                |
| Oct. 18 - Oct. 21, 2024     | ISRERM2024                                                                  |
|                             |                                                                             |
|                             |                                                                             |

# Doctor

## 

## 

Uncertainty Qualification | Model Updating

- Michael Beer, University of Hannover(Chair) 土木，不确定性
- David Moens 机械（高速相机，缺陷裂纹，机器视觉，不确定性）
- [sifeng-bi](https://pureportal.strath.ac.uk/en/persons/sifeng-bi) [BI Sifeng_北京理工大学留学生中心](https://isc.bit.edu.cn/schools/ae/knowingprofessors/abb4a9fee94d427f9fb572b1cf71b018.htm) 
  - https://scholar.google.com/citations?hl=nl&user=kt29JpQAAAAJ&view_op=list_works&citft=1&citft=2&citft=3&email_for_op=yuanqi053%40gmail.com&sortby=pubdate
- 

CSC，English要求（会议、考试or其他）


# 专利

- 确定性模型修正
- 区间模型修正+摄动法+椭球

