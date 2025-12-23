---
zotero-key: 5JEKED2M
zt-attachments:
  - "5"
title: "Stochastic Model Updating with Uncertainty Quantification: An Overview and Tutorial"
created: 2025-02-17 06:26:36
modified: 2025-08-07 03:04:10
tags:
  - University
  - of
  - Strathclyde
  - /done
  - ⭐⭐⭐⭐⭐
collections: 0Review  2 RC-MLP  1 Stochastic inverse FRF
year: 2023
publication: Mechanical Systems and Signal Processing
citekey: biStochasticModelUpdating2023
author:
  - Sifeng Bi
  - Michael Beer
  - Scott Cogan
  - John Mottershead
---
| Title        | "Stochastic Model Updating with Uncertainty Quantification: An Overview and Tutorial"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author       | [Sifeng Bi,Michael Beer,Scott Cogan,John Mottershead]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Organization | a Aerospace Centre of Excellence, Department of Mechanical and Aerospace Engineering, University of Strathclyde, Glasgow, **UK**  b Institute for Risk and Reliability, Leibniz University Hannover, Hannover, Germany  c Institute for Risk and Uncertainty, University of Liverpool, Liverpool, UK  d International Joint Research Center for Resilient Infrastructure & International Joint Research Center for Engineering Reliability and Stochastic Mechanics, Tongji University, Shanghai, China  e Department of Applied Mechanics, Femto-ST Institute, Besancon, France |
| Paper        | [Zotero pdf](zotero://select/library/items/5JEKED2M) [attachment](<file:///D:/Download/Zotero_data/storage/5Y239HYU/Bi%20%E7%AD%89%20-%202023%20-%20Stochastic%20Model%20Updating%20with%20Uncertainty%20Quantification%20An%20Overview%20and%20Tutorial.pdf>)<br><br>                                                                                                                                                                                                                                                                                                           |
| Project      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

<!-- more -->

## Background

**stochastic model updating主要可以分为频率派和贝叶斯派**
- 频率主义方法侧重于优化技术，以最小化现有测量和模型模拟之间的差异。
- 贝叶斯方法自然涉及更新中的不确定性处理。它还具有在存在非常罕见的测量数据的情况下捕获不确定性信息的优越性。因此,贝叶斯方法已被采用为随机模型更新中最受欢迎的技术之一
- 此外不精确概率技术也具有相当大的潜力，如
  - interval probabilities
  - evidence theory
  - info-gap theory
  - fuzzy probabilities

**stochastic model updating**包含几个关键步骤：
- **Feature definition**：有限元模型/实验输出的特征，可以是模态量(natural frequencies and mode shapes)，也可以是连续的量(例如时域系统响应或频率响应函数)。不同的输出特征比较需要不同的UQ指标
- Parameterisation and sensitivity analysis：参数化总是基于工程判断进行，参数化会产生大量可能对模型输出不确定或有意义的参数。随后灵敏度分析被开发为一种典型的技术，用于测量输入参数相对于输出特征的显着性，从而帮助选择下一步要校准的关键参数
- Surrogate modelling：代理有限元模型进行不确定性传播。The polynomial function，radial basis function，support vector machine，Kriging function，and neural network(BP or others)
- Parameter calibration：本质上是一个反向过程，以模拟和测量输出特征之间的差异为参考，并关注如何校准输入参数的原理和技术
  - 优化算法(eg: SSA)：将此任务描述为优化问题，输出差异被用来构建目标函数，通过搜索合适的参数值及其认识空间作为优化约束，该目标函数将被最小化(*基于灵敏度的优化方法*)
  - Bayes：其中参数的先验分布预计会根据现有测量的似然函数进行更新，并且更新的后验分布以更小的认识不确定性获得（*贝叶斯方法*）
- Model validation：模型验证是评估模型预测能力的重要步骤，验证准则
  - 模型应该预测现有的试验数据
  - 模型应该预测试验数据之外的独立数据（与修正参数不同的数据）
  - 模型应该预测物理系统的实际修改，在仿真模型中做同样的修改
  - 修正后的模型用在装配体的一个组件时，应该改进完全系统的装配体模型的预测

## Innovation

## Outlook

## Cases

### Tutorial example of the NASA UQ Challenge 2014


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250225100442.png)


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250225100434.png)

使用Bayesian model udpating修正4个参数的8个系数(均值、方差、相关系数、区间)

#### Forward uncertainty propagation

对于包含认知不确定性地参数，其参数分布地P-box可以用两组均值和方差确定

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250225100454.png)

通过Double-loop P-box propagation，可以得到输出特征地P-box

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250225100637.png)


#### Parameter calibration

对比different statistical distance-based likelihoods的修正结果
- 表格：参数名称、描述(参数的含义)、修正系数名称、初始区间、真实值、修正值（三种不同的距离指标）
- 图片：三张（不同距离指标） 修正系数的后验分布图
  - 讨论了距离指标对修正结果的影响及原因，

讨论Non-uniqueness solutions and output distributions
修正系数的后验PDFs中，会出现多峰问题，得到多个修正的系数，且验证后都可以计算得到很低的巴氏距离，并且对仿真得到的输出PDFs影响不大

#### Uncertainty reduction in the form of P-box reshaping

通过归一化修正系数的PDFs，然后分别在$\alpha=0.9,0.7,0.5,1$处截断，可以得到修正系数的不同区间(相较于原来的区间，缩小代表了认知不确定性的减小)，即得到参数的P-box，然后通过前向Double-loop P-box propagation得到输出的P-box

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250225102426.png)

### Practical example of an uncertain benchmark testbed

#### Design, experiment, and parameterisation of the model

设计
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250225102612.png)

试验
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250225102618.png)

仿真
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250225102628.png)

参数化
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250225102557.png)

#### Calibration results

Preliminary investigation of the measured natural frequencies
研究了试验测量的模态频率的特征和两两之间的关系

Influence of the likelihood function to the calibration results上个算例已经验证，不再赘述，但同样重要，未来可以进一步研究

基于even mean计算和weighted mean计算的似然函数(给不同的模态频率自定义赋予不同的权重)，并研究影响
- Likelihood based on even mean of the Bhattacharyya distances$BD_{even}=(BD_1+BD_2+BD_3+BD_4+BD_5)/5$
- Likelihood based on weighted mean of the Bhattacharyya distances$BD_{weight}=(BD_1+BD_2+BD_3+BD_4+2^*BD_5)/6$

表格：参数、描述、不确定性系数、原始区间、修正值(even mean和weighted mean)

图片：
- 修正后参数系数的PDFs
- 对比两种方法修正后的模态频率与试验测量之间的误差，以及与先验分布之间的差异 (分布图、二维散点图更直观)

Validation results
使用与之前修正不同的第七阶模态频率来验证模型修正的结果
表格对比：频率的均值和方差——实验值、even和weighted mean的似然函数

## Equation

### Bayesian model updating

[Transitional Markov Chain Monte Carlo Method for Bayesian Model Updating, Model Class Selection, and Model Averaging](../Method/Transitional%20Markov%20Chain%20Monte%20Carlo%20Method%20for%20Bayesian%20Model%20Updating,%20Model%20Class%20Selection,%20and%20Model%20Averaging.md)

$P(\theta|\mathbf{Y}_{exp})=\frac{P_L(\mathbf{Y}_{exp}|\theta)P(\theta)}{P(\mathbf{Y}_{exp})}$

似然函数：$P_L(\mathbf{Y}_{exp}|\theta)=\prod_{k=1}^{n_{exp}}P(\mathbf{y}_k|\theta)$，通过近似似然，构建与仿真-试验之间差异有关的函数$P_L(\mathbf{Y}_{exp}|\theta)\propto\exp\left\{-\frac{d\left(\mathbf{Y}_{exp},\mathbf{Y}_{sim}\right)^2}{\sigma^2}\right\}$

正则化因子：$P(\mathbf{X}_{exp})=\int P_L(\mathbf{Y}_{exp}|\theta)P(\theta)\mathrm{d}\theta$ 保证后验分布积分后的值为1。*该因子需要直接积分似然函数的显式的分布，是非常困难的*
为此提出MCMC方法，一步一步地获取中间PDFs，并最后收敛到后验PDFs：

$P^{(j)}(\theta)=P_L\left(Y_{\exp}|\theta\right)^{\beta_j}P(\theta)$


### Double-loop P-box propagation

先通过MC采样在$\alpha \in [0,1]$随机采样，对于不同类型的参数得到不同的 *random sets*。然后作为优化过程的约束条件，得到最小和最大输出特征对应的参数(*in random sets*)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250224110038.png)

前向传播方法对比

| 方法       | (本文) Monte-Carlo-Optimisation double-loop approach                         | (只用采样) Monte Carlo double-loop approach |
| -------- | -------------------------------------------------------------------------- | --------------------------------------- |
| 采样方式     | Aleatory采样+Epistemic采样                                                     | 只有Aleatory采样                            |
| 适用不确定性特征 | P-box（Aleatory+Epistemic混合）                                                | 随机分布（Aleatory）<br>区间量（Epistemic）        |
| 需要计算次数   | N次MC for $\alpha$ (each CDFs)<br>the number of CDFs N'<br>$total = N * N'$ | N次MC for (each CDFs or interval)        |

P-box是四条曲线构成的曲线：
- The CDF with minimum mean and minimum variance P(μmin, σ2min); 
- The CDF with maximum mean and minimum variance P(μmax, σ2min); 
- The CDF with minimum mean and maximum variance P(μmin, σ2max);
- The CDF with maximum mean and maximum variance P(μmax, σ2max).

为了直观理解:两组$\mu, \sigma$决定四个CDFs，进而决定一个P-box

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250224150124.png)


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250224135223.png)

- (a)为Beta分布的参数
- (b)为只有认知不确定性的参数，给定的一个区间量
- (c)为高斯分布的参数

 ***IMPORTANT of double-loop propagation***
 
之前的P-box of output features画错了(多条CDFs直接拟合)
错误做法：得到参数系数的后验分布之后($\mu,\sigma$的分布)，直接通过max和min选取分布的最大值和最小值。然后在最大和最小之间均匀等间隔采样$\mu, \sigma$，得到n组参数系数，即n个参数CDF，然后使用MC采样传播得到输出特征的n组CDF，用这n组CDF组成的曲线外轮廓当作P-box

正确做法：
得到参数系数的后验分布之后($\mu,\sigma$的分布)，然后归一化，并选取$\alpha$分别等于$[0.9,0.7,0.5,\dots]$处分布上的截断区间。以0.9为例，根据0.9对应的PDF上区间获得参数均值和方差的最大值和最小值(两组数据构成P-box)。

然后根据参数的P-box通过double-loop P-box propagation传播到output feeatures的P-box：首先通过MC采样$\alpha \in [0,1]$，得到N个概率，然后对每一个$\alpha$在参数P-box上得到对应的参数区间，然后通过求解优化问题得到output features的最大值和最小值，最后分别得到N个最大值和N个最小值 of output features，并且构成P-box的上下两条边界。同理可得0.7等时的系数PDF截断区间，进而得到P-box。

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250227164413.png)


### Statistic distance-based UQ metrics

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240412103257.png)

- 欧氏距离$d_E\big(\mathbf{Y}_{exp},\mathbf{Y}_{sim}\big)=\sqrt{\big(\overline{\mathbf{Y}_{exp}}-\overline{\mathbf{Y}_{sim}}\big)\big(\overline{\mathbf{Y}_{exp}}-\overline{\mathbf{Y}_{sim}}\big)^T}$
- 马氏距离$d_M\left(\mathbf{Y}_{exp},\mathbf{Y}_{sim}\right)=\sqrt{\left(\overline{\mathbf{Y}_{exp}}-\overline{\mathbf{Y}_{sim}}\right)\mathbf{C}^{-1}\left(\overline{\mathbf{Y}_{exp}}-\overline{\mathbf{Y}_{sim}}\right)^T}$
  - C is the “pooled” covariance matrix of both the simulation and measurement samples：$\mathbf{C}=\frac{(m-1)\mathbf{C}_{sim}+(n-1)\mathbf{C}_{exp}}{m+n-2}$
  - m and n are the numbers of simulated and measured sample
- 巴氏距离$d_B\big(\mathbf{Y}_{exp},\mathbf{Y}_{sim}\big)=-\log\bigg[\int_y\sqrt{P_{exp}(y)P_{sim}(y)}\mathrm{d}y\bigg]$
  - 当显式的PDF无法获取时，可以用PMF(Probability Mass Function)代替：$d_B\left(\mathbf{Y}_{exp},\mathbf{Y}_{sim}\right)=-\log\left[\sum_{i_m}^{n_{bin}}\cdots\sum_{i_1}^{n_{bin}}\sqrt{p_{exp}\left(b_{i_1,i_2,\cdots,i_m}\right)p_{sim}\left(b_{i_1,i_2,\cdots,i_m}\right)}\right]$，其中$n_{bin}$为从最大值到最小值之间划分的bin数量，通过离散化计算每个bin的样本数量来构造PMF。m为特征的维度(模态频率的个数)


> [how to calculate mahalanobis distance in pytorch? - Stack Overflow](https://stackoverflow.com/questions/65328887/how-to-calculate-mahalanobis-distance-in-pytorch)
> [概率分布之间的距离度量以及python实现 - 咖啡陪你 - 博客园](https://www.cnblogs.com/ltkekeli1229/p/15751619.html)
> [机器学习中不同的距离定义汇总（四：概率分布之间的距离）](https://blog.zhenyuan.cool/p/99b411b1823d6d8a/)

```
# 马氏距离
def mahalanobis(u, v, cov):
delta = u - v
m = torch.dot(delta, torch.matmul(torch.inverse(cov), delta))
return torch.sqrt(m)

# 巴氏距离
import numpy as np
p=np.asarray([0.65,0.25,0.07,0.03])
q=np.array([0.6,0.25,0.1,0.05])
BC=np.sum(np.sqrt(p*q))

#Hellinger距离：
h=np.sqrt(1-BC)
#巴氏距离：
b=-np.log(BC)
```