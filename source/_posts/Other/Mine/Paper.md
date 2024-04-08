# Model Updating

| Journal                                        | ISSN                 |
| ---------------------------------------------- | -------------------- |
| Mechanical Systems and Signal Processing       | 0888-3270            |
| Mechanics of Advanced Materials and Structures | 1537-6494            |
| Advances in Engineering Software               | 0965-9978            |
| International Journal of Computational Methods | 0219-8762            |
| Journal of Engineering Mechanics               | 0733-9399, 1943-7889 |


## A fast interval model updating method based on MLP neural network

ISRERM 会议 m202210465@xs.ustb.edu.cn | zhaoyanlin@ustb.edu.cn | ydb@ustb.edu.cn

**区间模型修正**
方法：基于 MLP 反向代理模型
算例：三自由度弹簧、钢板
论文：[Paper about ModelUpdating](Paper%20about%20ModelUpdating.md#Interval%20Model%20Updating)


### 会议流程

International Symposium on Reliability Engineering and Risk Management(ISRERM)
- October 18-21, 2024, Hefei, China
- 注册费、会议流程

## 专利

- 确定性模型修正
- 区间模型修正+摄动法+椭球

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

