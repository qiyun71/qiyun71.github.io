---
title: Case about ModelUpdating
date: 2024-03-07 21:26:57
tags:
  - ModelUpdating
categories: ModelUpdating
---

Model Updating算例

传统方法Python复现：[qiyun71/TraditionalModelUpdating: Traditional Model Updating by optim algo and loss function(or metrics)](https://github.com/qiyun71/TraditionalModelUpdating)
仿真数据训练：[qiyun71/NN-basedModelUpdating: NN-based Model Updating](https://github.com/qiyun71/NN-basedModelUpdating)
实验数据训练：[qiyun71/SSMU: Self-supervised Model Updating (SSMU) based on time-domain response data.](https://github.com/qiyun71/SSMU)

<!-- more -->

# Simple

## NASA challenge 2019

> [NASA Langley UQ Challenge on Optimization Under Uncertainty](https://uqtools.larc.nasa.gov/nasa-uq-challenge-problem-2020/)
> [Towards the NASA UQ Challenge 2019: Systematically forward and inverse approaches for uncertainty propagation and quantification - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327021007408?via%3Dihub)

- **Model Calibration** & Uncertainty Quantification of Subsystems
- Uncertainty Model Reduction
- Reliability Analysis
- Reliability-Based Design
- Model Updating and Design Parameter Tuning
- Risk-based Design

**Model Calibration:**

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240307224720.png)

如何根据目标的y(a,e,t) 校准不确定性参数 a & e


## IMAC2023

> [File:IMAC2023 FrameWing Nonlinear Substructuring Wiki.pptx - Dynamic Substructuring Focus Group Wiki](https://wiki.sem.org/wiki/File:IMAC2023_FrameWing_Nonlinear_Substructuring_Wiki.pptx)

>  [VIBES at IMAC Conference 2023 - VIBES.technology](https://www.vibestechnology.com/news/vibes-at-imac-conference-2023/) 

**The [International Modal Analysis Conference (IMAC)](https://sem.org/imac) is a yearly gathering of around 600 engineers, researchers and vendors active in the field of experimental mechanics. Specifically, it is seen as somewhat of the birthplace of Dynamic Substructuring technologies, as much of the pioneering research has been presented here.**

>  [Dynamic Substructuring Wiki - Dynamic Substructuring Focus Group Wiki](https://wiki.sem.org/wiki/Dynamic_Substructuring_Wiki) the SEM/IMAC Dynamic Substructuring Technical Division's Wiki.

### Dynamic substructuring

[Dynamic substructuring - Wikipedia](https://en.wikipedia.org/wiki/Dynamic_substructuring)

Dynamic Substructuring (DS) is an engineering tool used to model and analyse the dynamics of mechanical systems by means of its components or substructures. Using the dynamic substructuring approach one is able to analyse the dynamic behaviour of substructures separately and to later on calculate the assembled dynamics using **coupling procedures**. Dynamic substructuring has several advantages over the analysis of the fully assembled system:
- Substructures can **be modelled in the domain that is most appropriate**, e.g. experimentally obtained substructures can be combined with numerical models——从实验得到的子结构数据可以与仿真数据进行组合 
- **Large and/or complex systems** can be optimized on substructure level.
- **Numerical computation load** can be reduced as solving several substructures is computationally less demanding than solving one large system.
- Substructure models of different development groups can be shared and combined **without exposing the modelling details**.


![Two levels of domain decomposition in dynamic substructuring.|666](https://upload.wikimedia.org/wikipedia/commons/c/ca/Two_different_levels_of_domain_decomposition.svg)

## 结构参数

**Finite element thin wing modes**

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250110144922.png)


Thickness: 3.147mm

|          | Density $kg/m^3$ | Young's modulus Gpa | Poisson’s ratio |
| -------- | ---------------- | ------------------- | --------------- |
| Aluminum | 2708.3           | 70.38               | 0.33            |
| Steel    | 7850.0           | 210.0               | 0.3             |

**Finite element fuselage modes**

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250110144943.png)


Thickness: 12.7mm

|          | Density $kg/m^3$ | Young's modulus Gpa | Poisson’s ratio |
| -------- | ---------------- | ------------------- | --------------- |
| Aluminum | 2699.4           | 69.8                |  0.315          |

## Mass Spring System

The absolute value of the first component of the first eigenvector reflects some vibration information. The introduction of structural vibration modes as output responses will increase the difficulty of IMU.

**Numerical case studies: a mass-spring system**

| 结构参数         | Well-Separated | Close      | 动力学响应                 |
| ------------ | -------------- | ---------- | --------------------- |
| $m_{1}$(kg)  | 1              | 1          | $\omega_1^2(rad/s)^2$ |
| $m_{2}$(kg)  | 1              | 4          | $\omega_2^2(rad/s)^2$ |
| $m_{3}$(kg)  | 1              | 1          | $\omega_3^2(rad/s)^2$ |
| $k_{1}$(N/m) | [0.8, 1.2]     | 0          | $\|\varphi(1,1)\|$    |
| $k_{2}$(N/m) | [0.8, 1.2]     | [7.5, 8.5] |                       |
| $k_{3}$(N/m) | 1              | 0          |                       |
| $k_{4}$(N/m) | 1              | [1.8, 2.2] |                       |
| $k_{5}$(N/m) | [0.8, 1.2]     | [1.8, 2.2] |                       |
| $k_{6}$(N/m) | 3              | 1          |                       |



![massSpring.png|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/massSpring.png)


### 传统优化方法

#### Interval Uncertainty Propagation
将参数的不确定性区间传播到动力学响应的不确定性区间

***Interval perturbation***

> 参考：[Interval parameter sensitivity analysis based on interval perturbation propagation and interval similarity operator](https://hal.science/hal-04273667v1/document)

$\overline{\widehat{\boldsymbol{f}}}=F(\boldsymbol{\theta}^c)+\sum_{j=1}^N\frac{\boldsymbol{F}\left(\theta_j^c+\delta\theta_j\right)-\boldsymbol{F}\left(\theta_j^c\right)}{\delta\theta_j}\Delta\theta_j$
$\underline{\widehat{\boldsymbol{f}}}=F(\boldsymbol\theta^c)-\sum_{j=1}^N\frac{\boldsymbol{F}\left(\theta_j^c+\delta\theta_j\right)-\boldsymbol{F}\left(\theta_j^c\right)}{\delta\theta_j}\Delta\theta_j$

***Monte Carlo***

随机采样结构参数，然后经过前向仿真计算得到对应的动力学响应

***M&K matrix(FE calculation)***

M-K matrix (FE): $M\ddot{X} + KX = 0$

$M = \left( \begin{matrix}  m_{1} & 0 & 0\\ 0 & m_{2} & 0 \\ 0 & 0 & m_{3} \end{matrix} \right)$

$K = \left( \begin{matrix}  k_{1}+k_{4}+k_{6} & -k_{4} & -k_{6}\\ -k_{4} & k_{2}+k_{4}+k_{5} & -k_{5} \\ -k_{6} & -k_{5} & k_{3}+k_{5}+k_{6} \end{matrix} \right)$


$(-M \omega^{2}+K)A = 0$
$|-M \omega^{2}+K| = 0$

$\omega^{2} = M^{-1}K = Q\Sigma Q^{\top}$

$\omega_{1}^{2} = \Sigma(1,1)$
$\omega_{2}^{2} = \Sigma(2,2)$
$\omega_{3}^{2} = \Sigma(2,2)$
$|\varphi(1,1)| = Q(1,1)$

***FE surrogate model***

Response Surface Model (RSM)
- Well-separated modes
$$
\begin{aligned}
m_{1}& =1(\mathrm{kg}),\quad m_2=1(\mathrm{kg}),\quad m_3=1(\mathrm{kg}); \\
k_{3}& =k_4=1\mathrm{(N/m)},\quad k_6=3\mathrm{(N/m)}; \\
k_{1}& =k_2=k_5=[0.8,1.2]\text{(N/m).} 
\end{aligned}
$$



$$
\begin{aligned}\omega_1^2&=0.2840+0.3416k_1+0.4122k_2+0.0078k_5+0.0745k_1k_2+0.0011k_1k_5\\&-0.0014k_2k_5-0.0423k_1^2-0.0753k_2^2-0.0020k_5^2, \\
\omega_2^2&=1.6117+0.1249k_1+0.5882k_2+1.7402k_5-0.0735k_1k_2+0.1243k_1k_5\\&-0.0015k_2k_5-0.0021k_1^2+0.0748k_2^2-0.1871k_5^2, \\
\omega_3^2&=7.1036+0.5331k_1+0.0001k_2+0.2531k_5-0.0014k_1k_2-0.1247k_1k_5\\&+0.0025k_2k_5+0.0444k_1^2+0.0007k_2^2+0.1885k_5^2, \\
|\varphi(1,1)|&=0.5642-0.0894k_1+0.1060k_2+0.0171k_5+0.0082k_1k_2+0.0059k_1k_5\\&-0.0194k_2k_5+0.0009k_1^2-0.0150k_2^2-0.0012k_5^2.\end{aligned}
$$

- Close modes

$$
\begin{aligned}
m_{1}& =1(\mathrm{kg}),\quad m_2=4(\mathrm{kg}),\quad m_3=1(\mathrm{kg}); \\
k_{1}& =k_3=0\mathrm{(N/m)},\quad k_6=1\mathrm{(N/m)}; \\
k_{2}& =[7.5,8.5](\mathrm{N/m}),\quad k_4=k_5=[1.8,2.2](\mathrm{N/m}). 
\end{aligned}
$$

$$
\begin{aligned}\omega_1^2&=-0.0002+0.0830k_2+0.0839k_4+0.0842k_5+0.0186k_2k_4+0.0185k_2k_5\\&-0.0094k_4k_5-0.0046k_2^2-0.0325k_4^2-0.0325k_5^2,\\
\omega_2^2&=1.6103+0.0104k_2+1.0455k_4-0.0937k_5-0.0097k_2k_4+0.0055k_2k_5\\&+0.0094k_5+0.0042k_2^2+0.0396k_4^2+0.0005k_5^2, \\
\omega_3^2&=1.1103+0.0273k_2+0.0162k_4+1.1572k_5-0.0003k_2k_4-0.0165k_2k_5\\&+0.0104k_4k_5+0.0065k_2^2-0.0034k_4^2+0.0372k_5^2,\\
|\varphi(1,1)|&=0.6658+0.0125k_2-0.0988k_4+0.0496k_5-0.0062k_2k_4+0.0072k_2k_5\\&+0.0020k_4k_5-0.0005k_2^2+0.0190k_4^2-0.0170k_5^2.\end{aligned}
$$

#### Compare

| 区间传播方法                      | Interval Perturbation(First-order) | Monte Carlo(MC) |
| --------------------------- | ---------------------------------- | --------------- |
| M-K matrix (FE)             | 😊                                 |                 |
| Response Surface Model(RSM) |                                    |                 |

***well-separated modes***
- 使用M&K(FE)或者RSM，与Monte Carlo方法得到的响应区间相比，区间摄动法得到的$|\varphi(1,1)|$ 区间误差很大，$\Delta|\varphi(1,1)|$ 计算的偏小

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240603104442.png)

***close modes***
- (绿色)通过质量刚度矩阵(M&K)和蒙特卡洛法得到的响应区间在ws模式时可以很准确，但是在cl模式下，使用M&K与RSM相比有一定误差，主要是对$\omega _{2}^{2}$和$\omega _{3}^{2}$预测的不好
- (紫色)RSM和区间摄动法得到$|\varphi(1,1)|$的响应区间相较于(黑色)RSM和MC方法的误差还是很大

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240603104718.png)

### 基于MLP方法(Interval)

>  [qiyun71/MU_MassSpring](https://github.com/qiyun71/MU_MassSpring)

***Well-separated modes***

数据集生成：
- 在区间$[0,2]$内均匀生成10000组$k_1,k_2,k_5$
- 根据[Interval Identification of Structural Parameters Using Interval Deviation Degree and Monte Carlo Simulation](Interval%20Identification%20of%20Structural%20Parameters%20Using%20Interval%20Deviation%20Degree%20and%20Monte%20Carlo%20Simulation.md)，关于$\begin{aligned}\omega_1^2,\omega_2^2,\omega_3^2,\text{ and }|\varphi(1,1)|\end{aligned}$ 四个参数的二阶RSM(根据CCD(central composite design)生成15个samples)，得到10000组$\begin{aligned}\omega_1^2,\omega_2^2,\omega_3^2,\text{ and }|\varphi(1,1)|\end{aligned}$

目标：网络可以根据一组输入的$\begin{aligned}\omega_1^2,\omega_2^2,\omega_3^2,\text{ and }|\varphi(1,1)|\end{aligned}$得到对应的一组$k_1,k_2,k_5$

| 实验             | 初始区间(N/m)        | KP的*错误率*      | IRSM           | PF&RBF-NN | IOR&MC         | IDD&MC         | **本文方法**       |
| -------------- | ---------------- | ------------- | -------------- | --------- | -------------- | -------------- | -------------- |
| $[0.80, 1.20]$ | $k_1=[0.5, 1.5]$ | $[0.4,  0.0]$ | $[0.81, 1.20]$ | NAN       | $[0.79, 1.21]$ | $[0.80, 1.20]$ | $[0.80, 1.20]$ |
| $[0.80, 1.20]$ | $k_2=[0.5, 1.5]$ | $[0.8, 1.7]$  | $[0.80, 1.21]$ | NAN       | $[0.80, 1.20]$ | $[0.80, 1.19]$ | $[0.80, 1.20]$ |
| $[0.80, 1.20]$ | $k_5=[0.5, 1.5]$ | $[0.8, 1.7]$  | $[0.80, 1.20]$ | NAN       | $[0.80, 1.20]$ | $[0.80, 1.20]$ | $[0.80, 1.20]$ |
| ER             |                  |               |                | NAN       |                |                |                |
|                | $[37.5, 25]$     | $[0.4,  0.0]$ | $[1.3, 0]$     | NAN       | $[1.3, 0.8]$   | $[0, 0]$       | $[0, 0]$       |
|                | $[37.5, 25]$     | $[0.8, 1.7]$  | $[0, 0.8]$     | NAN       | $[0, 0]$       | $[0, 0.8]$     | $[0, 0]$       |
|                | $[37.5, 25]$     | $[0.8, 1.7]$  | $[0, 0]$       | NAN       | $[0, 0]$       | $[0,0]$        | $[0, 0]$       |
| **mean**       | $[37.5, 25]$     | $[0.7, 1.1]$  | $[0.4, 0.3]$   | NAN       | $[0.4, 0.3]$   | $[0, 0.3]$     | $[0, 0]$       |

***Close modes***

数据集生成：10000组$k_2,k_4,k_5$-->10000组$\begin{aligned}\omega_1^2,\omega_2^2,\omega_3^2,\text{ and }|\varphi(1,1)|\end{aligned}$

目标：网络可以根据一组输入的$\begin{aligned}\omega_1^2,\omega_2^2,\omega_3^2,\text{ and }|\varphi(1,1)|\end{aligned}$得到对应的一组$k_2,k_4,k_5$

| 实验           | 初始区间(N/m)        | KP的*错误率*     | IRSM           | PF&RBF-NN | IOR&MC         | IDD&MC         | **本文方法**       |
| ------------ | ---------------- | ------------ | -------------- | --------- | -------------- | -------------- | -------------- |
| $[7.5, 8.5]$ | $k_2=[6.5, 9.5]$ | $[0.6, 0.7]$ | $[7.55, 8.54]$ | NAN       | $[7.48, 8.50]$ | $[7.46, 8.52]$ | $[7.50, 8.50]$ |
| $[1.8, 2.2]$ | $k_4=[1.6, 2.4]$ | $[0.8, 1.0]$ | $[1.80, 2.19]$ | NAN       | $[1.80, 2.21]$ | $[1.80, 2.20]$ | $[1.80, 2.20]$ |
| $[1.8, 2.2]$ | $k_5=[1.5, 2.4]$ | $[0.4, 0.5]$ | $[1.80, 2.20]$ | NAN       | $[1.80, 2.21]$ | $[1.81, 2.20]$ | $[1.80, 2.20]$ |
| ER           |                  |              |                | NAN       |                |                |                |
|              | $[13.3, 11.8]$   | $[0.6, 0.7]$ | $[0.7, 0.5]$   | NAN       | $[0.3, 0]$     | $[0.5, 0.2]$   | $[0, 0]$       |
|              | $[11.1, 9.1]$    | $[0.8, 1.0]$ | $[0, 0.5]$     | NAN       | $[0, 0.5]$     | $[0, 0]$       | $[0, 0]$       |
|              | $[11.1, 9.1]$    | $[0.4, 0.5]$ | $[0, 0]$       | NAN       | $[0, 0.5]$     | $[0.6, 0]$     | $[0, 0]$       |
| **mean**     | $[11.8, 10.0]$   | $[0.6, 0.7]$ | $[0.2, 0.3]$   | NAN       | $[0.1, 0.3]$   | $[0.4, 0.1]$   | $[0, 0]$       |
|              |                  |              |                |           |                |                |                |


### 基于NN方法(Stochastic) 未完成

***Well-separated modes***

> [Interval Identification of Structural Parameters Using Interval Deviation Degree and Monte Carlo Simulation](Interval%20Identification%20of%20Structural%20Parameters%20Using%20Interval%20Deviation%20Degree%20and%20Monte%20Carlo%20Simulation.md)

| 结构参数    | 待修正参数            | 区间范围(均匀生成1000) |
| ------- | ---------------- | -------------- |
| $k_{1}$ | $\mu_{k_{1}}$    | $[0,2]$        |
|         | $\sigma_{k_{1}}$ | $[0,2]$        |
| $k_{2}$ | $\mu_{k_{2}}$    | $[0,2]$        |
|         | $\sigma_{k_{2}}$ | $[0.1,0.2]$    |
| $k_{5}$ | $\mu_{k_{5}}$    | $[0.1,0.2]$    |
|         | $\sigma_{k_{5}}$ | $[0.1,0.2]$    |

- 在区间$[0,2]$ 和 $[0.1,0.2]$内均匀生成1000组的均值与方差：$\mu_{k_{1}}$, $\sigma_{k_{1}}$, $\mu_{k_{2}}$, $\sigma_{k_{2}}$, $\mu_{k_{5}}$, $\sigma_{k_{5}}$
- 每一组均值与方差生成60组$k_{1}$, $k_{2}$, $k_{5}$
- 根据RSM计算得到60组$\begin{aligned}\omega_1^2,\omega_2^2,\omega_3^2,\text{ and }|\varphi(1,1)|\end{aligned}$

共需要计算60x1000次有限元模型(代理模型)

目标：网络可以根据一组60个输入$\begin{aligned}\omega_1^2,\omega_2^2,\omega_3^2,\text{ and }|\varphi(1,1)|\end{aligned}$得到对应的$k_1,k_2,k_5$三个参数的分布(用PDF曲线表示)
- 输入：60x4
- 输出：

***Close modes***


# Complex

## Steel Plate Structures

### 基于NN方法

Young’s modulus (E) and the shear modulus (G)

| 材料参数       | Nominal value |     |
| ---------- | ------------- | --- |
| 弹性模量E(GPa) | 205           |     |
| 剪切模量G(GPa) | 83            |     |

![图片1.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/%E5%9B%BE%E7%89%871.png)

***数据集生成***

| 材料参数   | 区间均值 | 区间半径 | 区间          |
| ------ | ---- | ---- | ----------- |
| E(GPa) | 205  | 15   | $[190,220]$ |
| G(GPa) | 83   | 6    | $[77,89]$   |

$$\begin{gathered}
f_{1} =13.1+0.2152E-0.01455G-0.0002813E^2-0.0004878E\cdot G+0.0006576G^2, \\
f_{2} =44.08+0.5145E-0.1333G+0.0002943E^2-0.004055E\cdot G+0.005588G^2, \\
f_{3} =52.85-0.1156E+1.445G+0.0006092E^2+0.0002019E\cdot G-0.005634G^2, \\
f_{4} =375.4-3.207E-0.4291G+0.02202E^2-0.009699E\cdot G+0.01383G^2, \\
f_{5} =79.55+0.255E+2.804G-0.0005059E^2-0.001249E\cdot G-0.009833G^2. 
\end{gathered}$$

在区间内生成10000组E和G，然后根据二阶RSM生成10000组的$f_1,f_2,f_3,f_4,f_5$

***修正结果***

| 实验  | 初始区间(GPa)     | IRSM            | PF&RBF-NN       | IOR&MC           | IDD&MC           | **本文方法**         |
| --- | ------------- | --------------- | --------------- | ---------------- | ---------------- | ---------------- |
| NAN | $E=[190,220]$ | $[196.5,203.6]$ | $[196.1,204.9]$ | $[196.2, 204.8]$ | $[196.2, 204.8]$ | $[196.8, 204.2]$ |
| NAN | $G=[77,89]$   | $[79.5,83.4]$   | $[79.5,83.3]$   | $[79.1, 83.7]$   | $[79.2, 83.6]$   | $[80.3, 84.2]$   |


根据修正后的E和G，得到相应的$f_1,f_2,f_3,f_4,f_5$，进行对比：

| 参数    | 实验                 | 初始区间(Hz)           | IRSM(**初始不同**)     | PF&RBF-NN         | IOR&MC             | IDD&MC             | **本文方法** |
| ----- | ------------------ | ------------------ | ------------------ | ----------------- | ------------------ | ------------------ | -------- |
| $f_1$ | $[42.66, 43.64]$   | $[42.12, 45.50]$   | $[ 42.87, 43.71]$  | $[42.82,43.82]$   | $[42.81, 43.81]$   | $[42.83, 43.82]$   |          |
| $f_2$ | $[118.29, 121.03]$ | $[116.16, 126.62]$ | $[118.45, 121.11]$ | $[118.28,121.38]$ | $[118.28, 121.38]$ | $[118.29, 121.38]$ |          |
| $f_3$ | $[133.24, 136.54]$ | $[131.48, 141.32]$ | $[133.58, 136.79]$ | $[133.60,136.73]$ | $[133.25, 137.05]$ | $[133.36, 136.96]$ |          |
| $f_4$ | $[234.07, 239.20]$ | $[227.77, 250.55]$ | $[232.78, 238.78]$ | $[232.28,239.03]$ | $[232.25, 239.10]$ | $[232.27, 239.08]$ |          |
| $f_5$ | $[274.29, 280.64]$ | $[269.10, 289.20]$ | $[273.45, 279.86]$ | $[273.39,279.73]$ | $[272.77, 280.36]$ | $[272.96, 280.20]$ |          |
| ER：   |                    |                    |                    |                   |                    |                    |          |
| $f_1$ |                    | $[1.27, 4.26]$     | $[0.5, 0.2]$       | $[0.37, 0.41]$    | $[0.35,0.39]$      | $[0.40, 0.41]$     |          |
| $f_2$ |                    | $[1.80, 4.62]$     | $[0.1, 0.1]$       | $[0.01, 0.29]$    | $[0.01,0.29]$      | $[0.00, 0.29]$     |          |
| $f_3$ |                    | $[1.32, 3.50]$     | $[0.3, 0.2]$       | $[0.27, 0.14]$    | $[0.01,0.37]$      | $[0.09, 0.31]$     |          |
| $f_4$ |                    | $[2.69, 4.74]$     | $[0.6, 0.2]$       | $[0.76, 0.07]$    | $[0.78,0.04]$      | $[0.77, 0.05]$     |          |
| $f_5$ |                    | $[1.89, 3.05]$     | $[0.3, 0.3]$       | $[0.33, 0.32]$    | $[0.55,0.10]$      | $[0.48, 0.16]$     |          |
| mean  |                    | $[1.79, 4.03]$     | $[0.4, 0.2]$       | $[0.35, 0.25]$    | $[0.34,0.24]$      | $[0.35, 0.24]$     |          |

#### 问题讨论

##### Surrogate Model 问题

***FE surrogate model （RSM）问题***

- IDD中的RSM

[Interval Identification of Structural Parameters Using Interval Deviation Degree and Monte Carlo Simulation](Interval%20Identification%20of%20Structural%20Parameters%20Using%20Interval%20Deviation%20Degree%20and%20Monte%20Carlo%20Simulation.md)，这篇论文的RSM出来的$f_4$有问题(比较明显)，其他的如$f_5$也有一点问题

`E = np.random.uniform(190.0, 220.0, 1000)`
`G = np.random.uniform(77.0, 89.0, 1000)`

```python
f1 = 13.1  + 0.2152*E - 0.01455*G - 0.0002813*E**2 - 0.0004878*E*G + 0.0006576*G**2
f2 = 44.08 + 0.5145*E - 0.1333 *G + 0.0002943*E**2 - 0.004055 *E*G + 0.005588 *G**2
f3 = 52.85 - 0.1156*E + 1.445  *G + 0.0006092*E**2 + 0.0002019*E*G - 0.005634 *G**2
f4 = 375.4 - 3.207 *E - 0.4291 *G + 0.02202  *E**2 - 0.009699 *E*G + 0.01383  *G**2
f5 = 79.55 + 0.255 *E + 2.804  *G - 0.0005059*E**2 - 0.001249 *E*G - 0.009833 *G**2
```

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240321214722.png)

- IRSM中的RSM

[An interval model updating strategy using interval response surface models](An%20interval%20model%20updating%20strategy%20using%20interval%20response%20surface%20models.md)，差的更多

```python
f1 = 77.624    - 0.0202366  *(E-40.905  )**2 + 0.010914*(G-2.342 )**2
f2 = -2214.965 + 0.002557626*(E+955.633 )**2 + 0.099410*(G-2.274 )**2
f3 = 218.882   - 0.00043976 *(E-181.515 )**2 - 0.084119*(G-28.413)**2
f4 = 45.397    + 0.137681   *(E+37.303  )**2 + 0.35152 *(G-2.201 )**2
f5 = 1887.265  - 0.000257232*(E-2387.009)**2 - 0.139010*(G-31.986)**2
```

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240322095004.png)


***解法：使用有限元模型***+不设置泊松比

- Solidworks建立钢板模型，保存为`.x_t`文件
- 使用Patran进行仿真，modes frequency和modes shape正常

| Modes    | mode1   | mode2   | mode3   | mode4   | mode5   |
| -------- | ------- | ------- | ------- | ------- | ------- |
| 模态频率(Hz) | 44.3433 | 122.654 | 136.691 | 241.305 | 279.993 |

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240427210132.png)

***问题***

- Nastran仿真计算，$N=10000, E \in [190,220], G \in [77,89]$ ，仿真结果：(模态频率与E完全线性相关，而与G无关，此外模态频率之间也高度正相关)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240427224559.png)

> [材料力学：如何证明弹性模量、泊松比和剪切模量之间的关系？G=E/2(1+μ) - 知乎](https://zhuanlan.zhihu.com/p/453054808)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240427225308.png)

***解法***

在Patran材料特性参数中只设置E和G(以及密度$\rho$)，让泊松比自动计算，计算出来的**模态频率正常**

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240503145843.png)


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240503145822.png)


##### 区间or随机问题

上述方法中，只要是用均匀分布生成数据的，都算是随机问题，而区间问题则需要等间隔进行生成，防止有的点没有采样到而导致的区间上下界有误差
**但是**，等间隔分布由于其不连续，也会丢失一些部分。均匀分布与等间隔分布各有优势和缺点，到底哪个更好，哪个更能准确地把区间问题表示出来？**应该根据结合实际的情况进行考虑**

MC仿真/区间不确定性传播：均匀分布生成数据，还是等间隔生成数据，哪一个能更好地得到输出特征响应的区间

~~My view：~~
- ~~采样量少的时候，等间隔生成的数据更好一点，因为随机在均匀分布中采样会出现很大的空白~~
- ~~采样量多的时候，均匀分布生成的数据更好一点，均匀分布可以更好的出现一些~~

均匀分布随机采样，会出现局部区域采样点密集，一些区域出现空白的情况，而等间隔采样则可以避免这个问题。
- **进行模型修正的时候，使用等间隔进行采样更好一点**。
- **但是在训练NN/构建逆代理模型时，使用随机采样得到的数据更好**

## Airplane Model

> [Stochastic Model Updating with Uncertainty Quantification: An Overview and Tutorial - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327023006921)

| 结构参数                                               | nominal value | 实验模态频率     | Interval with 30 times measurements |
| -------------------------------------------------- | ------------- | ---------- | ----------------------------------- |
| a half wingspan a/mm                               | 300           | $f_{1}$/Hz | [18.57, 20.96]                      |
| a wingtip chord b/mm                               | 25            | $f_{2}$/Hz | [38.24, 41.70]                      |
| a wing structural thickness T/mm                   | 1.2           | $f_{3}$/Hz | [84.54, 95.14]                      |
| Young’s modulus of fuselage/wing join $E_{1}$/GPa  | 70 铝合金        | $f_{4}$/Hz | [98.99, 109.25]                     |
| Young’s modulus of fuselage/tail joint $E_{2}$/GPa | 70            | $f_{5}$/Hz | [135.43, 145.13]                    |

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240307214120.png)
![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240307213826.png)

$\mu_a,\sigma_a,\mu_b,\sigma_b,E_1,E_2,\rho$ --> $f_1,f_2,f_3,f_4,f_5$ 前5阶固有频率
- $\mu_{a}\in [290,310]$ , $\sigma_{a} \in [0,5]$ (mm)
- $\mu_{b}\in [20,30]$ , $\sigma_{b} \in [0,5]$ (mm)
- $\rho \in [-1.0,1.0]$  a,b is Joint Gaussian, $\rho$为a和b两分布的相关系数
- $E_{1} \in [0.5,0.9]$ ($10^{11}Pa$) Young’s modulus of fuselage/wing join
- $E_{2} \in [0.5,0.9]$ ($10^{11}Pa$) Young’s modulus of fuselage/tail joint

![1-s2.0-S0888327023006921-gr18_lrg.jpg (2672×1394)|666](https://ars.els-cdn.com/content/image/1-s2.0-S0888327023006921-gr18_lrg.jpg)

模态试验：(数据大小: 3 x (6273, 52, 65) )
- 测量点 (DOF)：共52个——翼展上有40个、尾翼上12个点
- 试验产品：共30个不同sizes的飞机机翼
- FRFs频率范围：10~500Hz共有6273个frequency sample points
- FRFs： Displacement、Velocity、Acceleration FRFs

### 传统优化方法

[飞机模型修正方案](飞机模型修正方案.md)

- BP神经网络——FE代理模型
- SSA优化算法(/PSO...)
- UQ指标：区间、子区间、椭圆、子椭圆相似度

数据：a的均值0.3(m)固定，不同方差sigA来生成50组数，使用1000组$[0.002,0.0066]$(m)均匀生成的方差，一共生成了50000组a，然后根据相似三角形，得到b（b与a呈近似负线性相关）

```matlab
MUA=.469840228557587;sigA=0.005748;
n=50; %a
up1=normrnd(MUA,sigA,n,1);
up2=0.109211482107639-0.4151629366*(0.337217-up1+ 0.192931562662125);
up1 = (up1-0.169839009642601)*1000;
up2 = (0.109211482107639-up2)*1000;
```

- 根据50000组ab，利用Nastran生成50000组的$f_{1}\sim f_6$，然后训练BP(代理模型)
- 使用BP生成exp的$f_{1} \sim f_6$数据，n=1000, $\mu_{exp}$和$\sigma_{exp}$确定，大小为(6, n)
- **开始修正**：(对a进行修正，以固定$\mu$，修正$\sigma$为例)
  - 针对a参数设置一个$\sigma$的先验区间，比如大概为0.X，区间设置为(0, 1)
  - SSA寻优
    - 随机确定一个$\sigma$，与固定的$\mu$一起，生成ab的数据：(2, n)
    - 利用ab数据：(2, n)，使用BP生成仿真数据sim：(6, n)
    - 根据UQ指标将仿真数据sim与实验数据exp进行对比
    - **重复寻优直到结束**
  - 得到$\sigma_{updated}$ 然后跟实验$\sigma_{exp}$进行对比求得误差

修正$\mu$同理

### 基于MLP方法

论文中：[基于NN飞机算例修正思路](基于NN飞机算例修正思路.md)
$\mu_a,\sigma_a,\mu_b,\sigma_b,E_1,E_2,\rho$ 7个参数 --> $f_1,f_2,f_3,f_4,f_5$ 前5阶固有频率
- $\mu_{a}\in [290,310]$ , $\sigma_{a} \in [0,5]$ (mm)
- $\mu_{b}\in [20,30]$ , $\sigma_{b} \in [0,5]$ (mm)
- $\rho \in [-1.0,1.0]$  a,b is Joint Gaussian, $\rho$为a和b两分布的相关系数
- $E_{1} \in [0.5,0.9]$ ($10^{11}Pa$) Young’s modulus of fuselage/wing join
- $E_{2} \in [0.5,0.9]$ ($10^{11}Pa$) Young’s modulus of fuselage/tail joint

***数据生成结果1(废除)***

杨标生成：**(ab的方差不能生成得太大)** [飞机算例数据集生成](飞机算例数据集生成.md)
- $\mu_{a}\in [290,310]$ , $\sigma_{a} \in [1.217,4.049]$ (mm)
- $\mu_{b}\in [20,30]$ , $\sigma_{b} \in [1.025,2.101]$ (mm)
- $\rho \in [-1.0,1.0]$  a,b is Joint Gaussian, $\rho$为a和b两分布的相关系数
- $E_{1} \in [0.4,1.0]$ ($10^{11}Pa$) Young’s modulus of fuselage/wing join

共生成了1000组$\mu_a,\sigma_a,\mu_b,\sigma_b,E_1,\rho$，每组根据正态分布生成60个a、b样本，生成60个相同的$E_1$和$\rho$

1000组$\mu_a,\sigma_a,\mu_b,\sigma_b$
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240320204219.png)


**暂定m=1000，n=100**
$\mu_a,\sigma_a,\mu_b,\sigma_b,E_1,E_2,\rho$ 7个参数在各自的区间内均匀分布，随机抽取m=1000组
针对抽取的m中的每一组$\mu_a,\sigma_a,\mu_b,\sigma_b,E_1,E_2,\rho$ ：需要生成n=100组$a,b,E_1,E_2,$数据
- 生成两个相关系数为$\rho$的标准正态分布$X$和$Y$ [方法参考](https://blog.csdn.net/kdazhe/article/details/104599229) n = 100
  - **用联合分布生成a和b**
- 根据$\mu_a,\sigma_a,\mu_b,\sigma_b$ ，得到a和b一般正态分布$X_{a} \sim N(\mu_{a},\sigma^{2}_{a}),X_{b} \sim N(\mu_b,\sigma^{2}_b)$ n = 100
- $a,b,E_1,E_2$：其中a和b为相关的正态分布生成，$E_1,E_2$ 为100个重复的固定数
- 根据100组$a,b,E_1,E_2$，使用Nastran，生成100组 $f_1,f_2,f_3,f_4,f_5$

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240312161326.png)

总共需要使用Nastran进行$1,000 \times 100 = 100,000$次计算
按照每次计算花费10s计算，共需要100万 s = 278 h = 11.57 day

***数据生成结果2***
- $\mu_{a}\in [290,310]$ , $\sigma_{a} \in [0,5]$ (mm)
- $\mu_{b}\in [20,30]$ , $\sigma_{b} \in [0,5]$ (mm)
- $\rho \in [-1.0,1.0]$  a,b is Joint Gaussian, $\rho$为a和b两分布的相关系数
- 机翼厚度$T\in [1.1,1.2]$mm
- 连接件刚度$E_{1} \in [0.5,0.9]$ ($10^{11}Pa$) Young’s modulus of fuselage/wing join

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240611155141.png)

```
# exp设置
f1 tensor(16.9921) tensor(22.6861) 19, 21
f2 tensor(37.3678) tensor(46.0559) 38, 42
f3 tensor(77.1566) tensor(102.7040) 85, 95
f4 tensor(96.6111) tensor(117.1446) 100, 110
f5 tensor(123.9420) tensor(152.1165) 135, 145
```


***Method 1***

 **单次测量频率预测单组参数(反向代理模型——确定的)**
根据多组参数的值$\{a,b,E_{1},E_{2},\rho\}_{i=1}^{1e6}$来计算参数的均值和方差or区间边界

***Method 2***

**直接预测均值和方差or区间边界(反向代理模型——随机/区间)**

神经网络训练：
- 输入：5x100大小的数组（100组 $f_1,f_2,f_3,f_4,f_5$）
- 输出/标签标签：7x1的向量（$\mu_a,\sigma_a,\mu_b,\sigma_b,E_1,E_2,\rho$ ）

训练思路：输入5x100大小的数组通过FC(全连接层)计算得到中间向量，然后reshape成3通道图片，使用CNN处理图片，提取特征并解码为7x1的向量

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240312161313.png)


方法缺点：训练完成的NN，必须要输入固定大小的数组

### 问题讨论

对于小区间例如$[302.2234,302.2323]$预测的效果不好$[302.2415,302.2204]$

下界和上界会预测反，这是由于小区间的上界下界本来就相近，即使预测反或者预测得偏差一点，也不会有很大的loss

```
## a_label
tensor([
302.4267, 302.2234, 302.6144, 308.2171, 309.9898, 309.9898, 309.9898,
......
293.0252, 304.0647, 304.0647], device='cuda:0')

tensor([
302.4413, 302.2323, 302.6641, 308.2196, 309.9922, 309.9953, 309.9919,
......
293.0608, 304.1299, 304.1080], device='cuda:0')
```

```
## a_pred
tensor([
302.4215, 302.2415, 302.6063, 308.2065, 309.9922, 309.9915, 309.9943,
......
293.0485, 304.0324, 304.0781], device='cuda:0')

tensor([
302.4599, 302.2204, 302.7219, 308.2034, 309.9624, 309.9611, 309.9670,
......
293.2524, 304.1639, 304.0648], device='cuda:0')
```


~~**解法**: 换成网络去预测区间中心和区间半径，但是可能会出现大数吃小数的问题~~
**解法**: 取消小区间的情况，将小区间当作误差来处理

- $a\in [290,310]$
- $b\in [20,30]$
- $T\in [1.1,1.2]$mm

| a_lower | a_upper | b_lower | b_upper | T_lower  | T_upper  |
| ------- | ------- | ------- | ------- | -------- | -------- |
| 290,308 | 292,310 | 20,29.8 | 20.2,30 | 1.1,1.18 | 1.12,1.2 |

取消小区间后还是有问题：对T(mm)的修正误差很大
真正问题：**发现是数据的问题**(ab固定，T上下改变，输出的频率几乎不变)，**重新生成数据** --> 解决


## Satellite Model

>  [qiyun71/SatelliteModelUpdating: Satellite FE Model Updating](https://github.com/qiyun71/SatelliteModelUpdating)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231225211854.png)

主体结构6个参数为：
- **主弹性模量**$\theta_1$ 70Gpa，
- 主密度 $\theta_2$  ，密度2.7x $10^{3} kg/m^{3}$ (英文论文) or **适配器厚度** 1mm(本 1)
- **中心筒厚度**$\theta_3$ 2mm
- 底板厚度 $\theta_4$ 1mm
- **剪切板厚度**$\theta_5$ 2mm
- 顶板厚度 $\theta_6$ 2.5mm
  - 其中 $\theta_{4}、\theta_6$ 为常数，其他为待修正参数，$\theta_2$ 训练效果很差，直接去掉了，可能造成影响(本 1)
  - 其他参数：模型高 1250mm，中心筒直径 400mm，顶板边长 600mm，剪切板长 1000mm，宽 400mm

卫星模型选取了 11 个节点，并测量了水平 X 和竖直 Y 两个方向的加速度频响数据，频率范围为 ==0-50Hz==（实际使用30Hz），频率间隔为 0.5。得到最终输入网络的频响图像尺寸为 2×11×61，对应标签形状为 4×1。

FR 数据转三维数组(多通道图片)：

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231225212852.png)


### 传统优化方法

- 有限元方法获取数据集耗费时间长，使用一个代理模型来代替有限元计算模型。
- 根据随机样本 $X_{s}$ 通过代理模型得到模拟响应 $Y_{s}$，动力学实验得到实验响应 $Y_{e}(f_{1} ... f_{6})$
- 用子区间相似度计算模拟样本 $Y_{s}$ 和试验样本 $Y_{e}$ 之间的值，并作为目标函数。
- 用麻雀搜索算法将目标函数迭代寻优，得到修正后的均值 $\mu$ 和标准差 $\sigma$

目标函数：子区间相似度、标准差椭圆

### 基于NN方法

#### UCNN

***数据集生成与预处理***

数据集：
- 输入：测量的卫星模型加速度频响数据，将加速度频响数据经处理转化为频响图像
- 标签：选取的不确定性参数($\theta_1,\theta_2,\theta_3,\theta_5$)组成向量

**数据集生成**
- 先在一定范围内生成均匀生成 10000 组待修正参数
  - **弹性模量** 50.0~90.0GPa
  - ~~**适配器厚度** 0.50~1.50mm~~
  - ~~**中心筒厚度** 1.00~3.00mm~~
  - ~~**剪切板厚度** 1.00~3.00mm # 本科生 1 数据集中生成的是 0.50~1.50mm~~
  - 顶板厚度 1.50~3.50mm

```markdown
/data/train 训练数据分为25组，每一组400条，共10000次仿真数据
- fre-400-1
  - sita1.xlsx #400x1
  - sita2.xlsx
  - ...
  - sita6.xlsx
  - xiangying.xlsx #80800x11 频响数据，每101×2行 为一条，对应0-50Hz 101个频率点x、y两个方向的数据. 从第0行 (x方向)开始, 偶数列为x方向,奇数列为y方向
- fre-400-2
- fre-400-3
- fre-400-4
- ...
- fre-400-25

/data/test 400条测试集，用于测试网络的精度
- fre-400
  - sita1.xlsx #400x1
  - sita2.xlsx 
  - ...
  - sita6.xlsx
  - xiangying.xlsx #80800x11
```


**卫星算例.bdf** 不同结构参数生成结构特征量FR

```bdf file
$ Elements and Element Properties for region : Shear_Panels
PSHELL   1       1      .003     1               1

- 36  行 .003 Shear_Panels 厚度 theta5
- 429 行 .002 Central_Cylinder 厚度 theta3
- 666 行 .001 Adapter 厚度 theta2 本来应该是密度2.7
- 723 行 .002 Upper_platform 厚度 theta6
- 864 行 .001 Lower_platform 厚度 theta4
- 1020行 7.   mat_N 弹性模量  theta1  
- 1023行 7.   mat_CC 弹性模量  theta1  
- 1026行 7.   mat_L 弹性模量  theta1  
- 1029行 7.   mat_SP 弹性模量  theta1  
- 1032行 7.   mat_U 弹性模量  theta1  
- 主弹性模量不包括 mat_A 适配器的材料属性
```

- **主弹性模量**$\theta_1$ 70Gpa，
- **主密度** $\theta_2$  ，密度2.7x $10^{3} kg/m^{3}$ (英文论文) or 适配器厚度 1mm(本 1)
- **中心筒厚度**$\theta_3$ 2mm
- 底板厚度 $\theta_4$ 1mm
- **剪切板厚度**$\theta_5$ 2mm
- 顶板厚度 $\theta_6$ 2.5mm

***网络结构***

UCNN 单向卷积

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231227133859.png)

***评价指标***

测试集错误率：$错误率 = \frac{|预测值-标签GT|}{标签GT}$

#### Improvement(Other NN)

***数据集生成***


**将excel数据转换成处理更快的numpy数组**
处理后训练集包含两类npy文件:
- `train_FRdata{i}.npy` 存储FRF数据 --> 400条数据,共25组 (其中一组损坏,实际24组) ==> (9600, 2, 61, 11)
- `label_{i}.npy` 存储 参数$\theta$ 数据 --> 400条数据,共25组 ==> (9600, 4)
测试集包含: `test_FRdata.npy` (400, 2, 61, 11) 与 `label.npy` (400, 4)
试验集包含: `test_FRdata.npy` (1000, 2, 61, 11) 与 `label.npy` (1000, 4) 

excel： n为data sets，训练集和测试集都是25个文件(25个sita1.xlsx)，每个文件中n为400. 试验集只有一个文件，n为1000
- **sita1.xlsx** 一列，n行
- **sita2.xlsx**
- **sita3.xlsx**
- sita4.xlsx
- **sita5.xlsx**
- sita6.xlsx
- **xiangying.xlsx**  80800行(400x101x2) 11列

npy: 
- test_FRdata.npy (data sets, 2, 61, 11) xy两方向、前61个频率点、选取的11个测量参考节点 
- label.npy (data sets, 4)

|              | $\theta_{1}$                    | $\theta_{2}$            | $\theta_{3}$            | $\theta_{4}$      | $\theta_{5}$            | $\theta_{6}$      |
| ------------ | ------------------------------- | ----------------------- | ----------------------- | ----------------- | ----------------------- | ----------------- |
| 物理参数         | 主弹性模量 E                         | Adapter 厚度              | Central_Cylinder 厚度     | Lower_platform 厚度 | Shear_Panels 厚度         | Upper_platform 厚度 |
| 名义值          | 70Gpa                           | 1mm                     | 2 mm                    | 1 mm              | **2 mm**                | 2.5 mm            |
| In Paper     | 70Gpa                           | 1mm                     | 2 mm                    | 2 mm 审稿后修改为1      | **1 mm**                | 2.5 mm            |
| 训练集          |                                 |                         |                         |                   |                         |                   |
| Excel        | $7 (\times 10^{10}Pa)$          | 0.001 (m)               | 0.002 (m)               | 0.001 (m)         | **0.001 (m)**           | 0.0025 (m)        |
| Code(python) | $7 (\times 10^{10}Pa)$          | 1 (mm)                  | 2 (mm)                  | No                | 1 (mm)                  | No                |
| 训练集 范围       | $[5.0, 9.0] (\times 10^{10}Pa)$ | $[0.5, 1.5] (mm)$       | $[1.0, 3.0] (mm)$       |                   | $[0.5, 1.5] (mm)$       |                   |
| 试验集          |                                 |                         |                         |                   |                         |                   |
| 试验数据 范围      | $E\sim N(7, 0.3^2)$             | $T_{1}\sim N(1, 0.2^2)$ | $T_{2}\sim N(2, 0.4^2)$ | $T_{4}=0.001$     | $T_{3}\sim N(1, 0.2^2)$ | $T_{5}=0.0025$    |
|              |                                 |                         |                         |                   |                         |                   |

数据集详细记录日志，data/
- FE：生成数据集的matlab程序，需要调用nastran
- test：测试网络精度数据集（均匀）
- test_1：Exp实验数据(Target目标数据)，测试网络修正结果的数据集（正态）
- test_1_pred：根据**test_1**预测的参数结果，输入nastran得到FR
- test_1_pred_norm：根据**test_1归一化后**预测的结果，输入nastran得到FR
- train：训练网络数据集（均匀）
- train_npy、test_npy、test_1_npy、test_1_pred_npy：excel转为npy格式
- train_npy copy：train转为npy时选取的频率范围为0~50Hz，其他为30Hz，用于绘制数据处理例子的流程图
- train_npy_24_3theta、test_npy_24_3theta为本科生的修正3个参数的数据

将FRF归一化后，主要数据文件夹： data/
- train_npy_norm 训练集 --> 训练网络
- test_npy_norm 测试集 --> 测试网络的训练效果
- test_1_npy_norm??? 实验集/目标集 --> 验证网络的校准效果(model calibration)
- test_1_npy??? 试验集没有归一化，为了最后的模型验证

添加噪声
- test_1_npy_norm1000_noise？ --> Trained NN --> /pred.npy --> FEA --> test_1_pred_norm_npy1000_noise?
- test_1_npy1000_noise？(Target without normalization)  <-->  test_1_pred_norm_npy1000_noise? (Calibrated)

***网络结构***

```ad-tldr UCNN 
**UCNN**

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231227133859.png)

**MLP**

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240111150150.png) 

**MLP_S**
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240111150239.png) 

**VGG16**
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240111145858.png) 

**ResNet50**

![resnet.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/picturesresnet.png)
```


***实验记录***

```ad-info
20231227-111217: 10, 0.9 UCNN
20231227-111738: 25, 0.99 UCNN
20231227-160116: 25, 0.99 MLP 4x128
20231227-160651: 25, 0.99 UCNN 3.5min
20231227-162720: 25, 0.99 MLP 8x256
20231227-163116: 25, 0.99 MLP 8x256 **400** epoch 4.0min
# 20240102-164844: 25, 0.99 MLP_S 14个4x256的MLP 200 epoch 9min
# 20240102-170301: 25, 0.99 MLP_S 14个2x256的MLP 200 epoch 5.6min
20240102-203115: 25, 0.99 MLP_S 14个2x256的MLP **400** epoch 15.1min
20240102-205253: 25, 0.99 MLP_S 14个4x256的MLP **400** epoch 23.27min
20240102-211805: 25, 0.99 UCNN **400** epoch 7min
# 20240111-160514: 25, 0.99 VGG11 200 epoch (由于图片太小使用VGG11)
20240111-162059: 25, 0.99 VGG11 400 epoch 26.34 min
20240113-190506: 25, 0.99 ResNet50 400 epoch 36.1 min

# 测试两个是不是过拟合了，error rate随epoch变化趋势
20240115-092119: 25, 0.99 UCNN 1000 epoch
20240115-111929: 25, 0.99 VGG11 1000 epoch

# 24x400 数据 --> theta1,3,5
20240116-142116: 25, 0.99 VGG11 400 epoch
20240116-153711: 25, 0.99 UCNN 400 epoch

# 24x400 数据 --> theta1,2,3,5
## 25, 0.99 | 400 epoch
@20240116-164606_ucnn	
@20240116-170957_vgg	
@20240116-180846_resnet	
@20240116-193319_mlp	
@20240116-194754_mlp_s

## 25, 0.99 | 1000 epoch
@20240116-213751_mlp	
@20240116-222413_ucnn	
@20240116-231550_vgg	
@20240117-014321_mlp_s	
@20240117-065536_resnet

## 25, 0.99 | 500 epoch
@20240118-232440_ucnn
@20240118-234300_vgg
@20240119-005355_mlp
@20240119-010645_mlp_s
@20240119-021612_resnet

## 50, 0.99 | 500 epoch
@20240119-225843_ucnn
@20240119-231920_vgg
@20240120-003157_mlp
@20240120-004544_mlp_s
@20240120-015533_resnet

## 100, 0.99 | 500 epoch
@20240120-171537_ucnn
@20240120-175110_mlp
@20240120-180848_mlp_s
@20240120-195338_vgg
@20240120-220032_resnet

## 200, 0.99 | 500 epoch
@20240120-234329_ucnn
@20240121-000302_mlp
@20240121-001554_mlp_s
@20240121-012709_vgg
@20240121-023818_resnet

## 300, 0.99 | 500 epoch
@20240121-040844_ucnn
@20240121-042747_mlp
@20240121-044111_mlp_s
@20240121-055239_vgg
@20240121-070346_resnet
``` 

不论 MLP 还是 UCNN，中间 sita3 预测的误差都很大

```ad-info
(satellite) PS D:\0Proj\ModelUpdating\satellite_UCNN> python run.py --test --resume outputs\@20231227-163116_mlp\400_mlp.pth --net mlp   
error_rate=:0.05787282592434471=(0.19222232587635518/3.321460855007172)
=====================
error_rate_each0=:0.04022682424495547=(0.28112122416496277/6.9884021282196045)
error_rate_each1=:0.12927590329299216=(0.25523426607251165/1.9743375182151794)
error_rate_each2=:0.04024536597068831=(0.040311464481055735/1.0016423881053924)

(satellite) PS D:\0Proj\ModelUpdating\SatelliteModelUpdating> python run.py --test --resume  outputs\@20240102-203115_mlp_s\400_mlp_s.pth --net mlp_s
error_rate=:0.03934166444889552=(0.13067179843783377/3.321460855007172)
=====================
error_rate_each0=:0.028448767469817206=(0.1988114271312952/6.9884021282196045)
error_rate_each1=:0.0800526909024898=(0.15805103108286858/1.9743375182151794)
error_rate_each2=:0.03509528007871003=(0.03515292014926672/1.0016423881053924)

(satellite) PS D:\0Proj\ModelUpdating\SatelliteModelUpdating> python run.py --test --resume  outputs\@20240102-205253_mlp_s\400_mlp_s.pth --net mlp_s # 过拟合了，训练集上的损失比较小，但是测试集上的loss比较大
error_rate=:0.03124857323569491=(0.10379091277718544/3.321460855007172)
=====================
error_rate_each0=:0.01964181444371989=(0.13726489786058665/6.9884021282196045)
error_rate_each1=:0.07125288617244709=(0.14067724645137786/1.9743375182151794)
error_rate_each2=:0.03337576539351412=(0.03343058135360479/1.0016423881053924)

(satellite) PS D:\0Proj\ModelUpdating\SatelliteModelUpdating> python .\run.py --test --resume outputs\@20240102-211805_ucnn\400_ucnn.pth
error_rate=:0.02647251309147154=(0.08792741596698761/3.321460855007172)
=====================
error_rate_each0=:0.022187629130279763=(0.15505607463419438/6.9884021282196045)
error_rate_each1=:0.03919995023153533=(0.07739393245428801/1.9743375182151794)
error_rate_each2=:0.031280856565034834=(0.03133223187178373/1.0016423881053924)

(satellite) PS D:\0Proj\ModelUpdating\SatelliteModelUpdating> python .\run.py --net vgg --test --resume outputs\202401\@20240111-162059_vgg\400_vgg.pth
error_rate=:0.02620232199857664=(0.0870299868285656/3.321460855007172)
=====================
error_rate_each0=:0.016490224581949674=(0.11524032056331635/6.9884021282196045)
error_rate_each1=:0.05937809189634055=(0.11723239459097386/1.9743375182151794)
error_rate_each2=:0.02857031368553574=(0.028617237228900194/1.0016423881053924)

(satellite) PS D:\0Proj\ModelUpdating\SatelliteModelUpdating> python .\run.py --test --resume outputs\202401\@20240113-190506_resnet\400_resnet.pth --net resnet
error_rate=:0.12867085358744207=(0.42737520337104795/3.321460855007172)
=====================
error_rate_each0=:0.1063032449177=(0.7428898230195046/6.9884021282196045)
error_rate_each1=:0.22810707873746822=(0.4503603637218475/1.9743375182151794)
error_rate_each2=:0.08872962392607951=(0.08887535240501165/1.0016423881053924)
```



### 问题讨论

[基于NN的卫星算例问题](基于NN的卫星算例问题.md)

- 根据目标参数expT的均值(7,1,2,1)和标准差(0.3,0.2,0.4,0.2)，生成服从正态分布的m组exp数据
  - 生成的数据根据torch.mean和torch.std再计算会有一定偏差，比如表中正态分布生成m=100组exp数据和m=1000组exp数据，1000组数据经过再次计算得到的均值和标准差，相较于100组偏差更小
- 对m组的实验FR数据进行修正，得到m组修正后的参数，然后根据torch.mean和torch.std再计算得到均值和标准差
  - 论文中是跟原始的（用于生成分布的）均值和标准差进行对比，但是实际应该与再计算的均值和标准差进行比较才更准确


**使用model.eva()的结果**：

|                | expT | 100组exp    | pred1  | pred2  | pred3  | 1000组exp   | pred1  | pred2  | pred3  |
| -------------- | ---- | ---------- | ------ | ------ | ------ | ---------- | ------ | ------ | ------ |
| $\mu_E$        | 7    | **6.9781** | 6.9613 | 6.9696 | 6.9764 | **6.9981** | 6.9762 | 6.9833 | 6.9874 |
| $\mu_{T_1}$    | 1    | **1.0415** | 1.0320 | 1.0368 | 1.0120 | **1.0061** | 0.9951 | 1.0041 | 1.0197 |
| $\mu_{T_2}$    | 2    | **2.0619** | 2.0641 | 2.0557 | 2.0562 | **2.0008** | 2.0005 | 1.9978 | 1.9730 |
| $\mu_{T_3}$    | 1    | **1.0046** | 1.0055 | 1.0031 | 1.0062 | **0.9965** | 0.9968 | 0.9941 | 0.9940 |
| $\sigma_{E}$   | 0.3  | **0.3235** | 0.3385 | 0.3333 | 0.3436 | **0.2940** | 0.3263 | 0.3161 | 0.3219 |
| $\sigma_{T_1}$ | 0.2  | **0.1961** | 0.1935 | 0.1899 | 0.1731 | **0.1905** | 0.1942 | 0.1884 | 0.1861 |
| $\sigma_{T_3}$ | 0.4  | **0.3903** | 0.3965 | 0.3904 | 0.3790 | **0.3828** | 0.3918 | 0.3858 | 0.3789 |
| $\sigma_{T_3}$ | 0.2  | **0.1954** | 0.1959 | 0.1928 | 0.1916 | **0.1932** | 0.1928 | 0.1912 | 0.1899 |
| ER_T(%)        | 0    | 2.82       | 3.31   | 3.67   | 5.30   | 1.93       | 2.31   | 2.56   | 3.58   |
| ER(%)          |      | 0          | 1.14   | 1.07   | 3.26   | 0          | 2.12   | 1.40   | 2.21   |
|                |      |            |        |        |        |            |        |        |        |
|                |      |            |        |        |        |            |        |        |        |

说明
- pred1：归一化处理FR数据、500epoch训练结果
- pred2：归一化处理FR数据、1000epoch训练结果
- pred3：不归一化，500epoch训练结果
- ER\_T(%) = $\frac{1}{8} \sum\limits_{i=1}^8 \frac{|pred_i-expT_i|}{expT_i}$ x100 (%)，**论文中关注的是这个指标**
- ER(%) = $\frac{1}{8} \sum\limits_{i=1}^8 \frac{|pred_i-exp_i|}{exp_i}$ x100 (%)，**实际应该是这个指标更准确一点**

由平均误差，可以看出：
1. **归一化处理**pred1比没有归一化处理pred3，训练的网络精度高~~（主要是方差精度，excel可以看出）~~
2. **epoch** 越多，ER越低(这是正常的)，但是ER_T会更高，这是由于ER_T是(pred)与生成分布时地均值和标准差(expT)对比的，exp与expT的误差与pred与exp的误差累积了(两个随机相互影响)
3. **数据量m**越多，ER_T更低，但是ER反而会更高(未归一化pred3反而更低)，**此外数据量多了还有一个问题**：

这是训练集 label（这副图的exp和updated是同一个变量，两者都是label）
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240308143958.png)

使用测试集测试网络训练的精度（均匀分布生成的测试集）
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240308144116.png)

实验数据得到修正结果(m=1000)（正态分布）
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240308144229.png)

实验数据(m=100)，数据量少的时候，这种特征还不明显
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240308144522.png)


更加明显，$T_1$和$T_3$集中在了(等间隔0.1的位置)，在微小变动时，
![1709889649879.jpg|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/1709889649879.jpg)

**问题**：$T_3$预测的分布规律奇怪，虽然也是正态分布，但是大多数值集中在了1.0、1.1等间隔0.1的位置，$T_1$也有类似情况但是没有$T_{3}$这么明显。
**分析**：神经网络对$T_3$不敏感，网络预测将$T_3$小数点后两位的数省略了（只识别出了1.0、1.1mm等规律的值），且这一现象与**是否对FR归一化**无关，没有FR归一化的预测也有这个现象：

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240308165855.png)

***猜想1（错误）***

与**结构参数归一化**有关，将FR和结构参数，也就是NN的输入和输出都归一化后，结果：

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240308201527.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240308201704.png)


***猜想2（错误）***

训练过程就识别不到其他位置的参数了，训练的1个epoch，发现除了边缘无法很好预测，其他地方也很满（中间有的地方也很空，但不会出现聚集在特定值的现象）

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240308203251.png)


***猜想3（错误）***

可能FR数据生成有问题，数据生成根据4个参数的均值和方差，生成1000组正态分布数据，然后根据1000组四参数，使用Nastran生成FR数据
可能当改变剪切板厚度$T_3$时，生成的FR数据过于相似，例如将$T_{3}=1.12$或者$T_{3}=1.13$等生成的FR数据跟$T_{3}=1.1$生成的FR数据近似，NN分辨不出来

400组数据的测试集中也有这个现象：
![image.png|222](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240309164056.png)

   
***真正原因***

在test和exp的时候，添加了`model.eval()`代码，会将网络中的BN层和Dropout层关闭（训练时是开启的），(test数据batchsize是直接设置成整个test dataset的大小，可以)不使用eval，相当于在训练集上过拟合了以后，可以更好地对先验区间中的实验数据进行更好的预测

**不使用**model.eval()的结果：

|                | expT | 100组exp    | pred1  | pred2  | pred3  | 1000组exp   | pred1  | pred2    | pred3  |
| -------------- | ---- | ---------- | ------ | ------ | ------ | ---------- | ------ | -------- | ------ |
| $\mu_E$        | 7    | **6.9781** | 6.9893 | 6.9781 | 6.9948 | **6.9981** | 6.9983 | 6.9962   | 7.0011 |
| $\mu_{T_1}$    | 1    | **1.0415** | 1.0336 | 1.0435 | 1.0505 | **1.0061** | 0.9921 | 0.9995   | 1.0215 |
| $\mu_{T_2}$    | 2    | **2.0619** | 2.0647 | 2.0583 | 2.042  | **2.0008** | 1.9988 | 1.9951   | 1.9789 |
| $\mu_{T_3}$    | 1    | **1.0046** | 1.0045 | 1.0033 | 1.0036 | **0.9965** | 0.9966 | 0.9953   | 0.9963 |
| $\sigma_{E}$   | 0.3  | **0.3235** | 0.3673 | 0.344  | 0.3365 | **0.2940** | 0.3299 | 0.3232   | 0.323  |
| $\sigma_{T_1}$ | 0.2  | **0.1961** | 0.1956 | 0.1937 | 0.1947 | **0.1905** | 0.1957 | 0.1919   | 0.1925 |
| $\sigma_{T_3}$ | 0.4  | **0.3903** | 0.3977 | 0.3965 | 0.3885 | **0.3828** | 0.3934 | 0.3919   | 0.3852 |
| $\sigma_{T_3}$ | 0.2  | **0.1954** | 0.1988 | 0.1914 | 0.1928 | **0.1932** | 0.1925 | 0.1925   | 0.1926 |
| ER_T(%)        | 0    | 2.82       | 4.13   | 3.86   | 3.61   | 1.93       | 2.34   | **2.30** | 2.80   |
| ER(%)          |      | 0          | 2.31   | 1.46   | 1.09   | 0          | 2.45   | 1.81     | 1.82   |

论文关注的ER_T，其中1000组exp的pred2效果更好(归一化处理FR数据、1000epoch训练结果)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240309201551.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20240309201706.png)

#### 程序问题

matlab 生成FRFs的程序问题，并不是11个节点的加速度幅值

```f06
      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3
0          118      G     -1.893191E-02   2.744646E-02  -1.148516E-02  -5.853226E-02  -1.082212E-04   1.105988E-01
                           4.502489E-07  -6.520212E-07   2.733072E-07   1.392180E-06   2.492752E-09  -2.660447E-06
0          237      G      1.893236E-02   2.744877E-02   1.148540E-02  -5.853223E-02   1.081836E-04   1.106056E-01
                          -4.502597E-07  -6.520763E-07  -2.733128E-07   1.392179E-06  -2.491843E-09  -2.660618E-06
0          255      G      4.782312E-08   1.635185E-02   2.330007E-08  -5.961377E-02  -2.843844E-07   9.934657E-02
                          -1.020500E-12  -3.889889E-07  -4.840865E-13   1.414843E-06   5.956261E-12  -2.356463E-06
0          381      G      4.910261E-10   5.798578E-02  -1.367990E-09   1.038435E+00   6.459272E-07   9.908993E-02
                          -9.960504E-15  -1.379422E-06   2.324609E-14  -1.986041E-05  -1.446684E-11  -2.354878E-06
0          416      G      3.527358E-09  -6.694251E-01   2.848346E-09  -7.681564E-01  -1.106894E-08  -2.531782E+00
                          -5.386538E-14   1.314468E-05  -6.025993E-14   1.584632E-05   2.442403E-13   5.072122E-05
0          446      G     -6.481036E-09   1.164889E-01   5.534058E-09  -1.548480E-01  -5.967503E-08   1.012482E-01
                           1.533396E-13  -2.771816E-06  -1.133209E-13   3.226627E-06   1.407623E-12  -2.393049E-06
0          521      G      6.857802E-09  -5.797762E-02  -5.915352E-09  -2.194655E+00  -1.031755E-06   9.962211E-02
                          -1.611111E-13   1.379118E-06   1.601390E-13   6.070103E-05   2.778342E-11  -2.378132E-06
0          556      G     -2.086969E-07   1.367283E+00  -1.148249E-09   1.063863E+00   1.623497E-07  -4.669222E+00
                           5.754569E-12  -3.771298E-05   2.932038E-14  -2.973616E-05  -4.530125E-12   1.288144E-04
0          587      G      2.343825E-02  -1.353126E-02   1.413584E-02  -1.120405E-01  -8.874521E-02   1.027448E-01
                          -5.575564E-07   3.218754E-07  -3.363756E-07   2.844023E-06   2.379223E-06  -2.463951E-06
0          728      G      4.197688E-11   6.445141E-03  -5.843937E-09  -5.821381E-02  -9.521039E-08   9.667463E-02
                          -8.090375E-16  -1.533222E-07   1.454717E-13   1.387308E-06   1.444107E-12  -2.299757E-06
0          827      G     -6.594866E-09   5.828727E-02  -9.611829E-09  -5.830408E-02  -9.875006E-09   9.698171E-02
                           1.556223E-13  -1.386983E-06   2.443410E-13   1.387674E-06   3.523319E-13  -2.307504E-06
```

matlab读取.f06文件时，获取T1列，然后存入motai.txt中的一行

```matlab
for i=1:21 
   line=fgetl(fid);
   wlc=[wlc;str2double(line(27:40))];
end
% 然后存入motai.txt
fclose(fid);
fid=fopen('motai.txt','a');
fprintf(fid,'%3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d\r\n',wlc);
fclose(fid);

A =importdata('motai.txt');
% a=A(1:202,[3 19 11 7 10 13 15 1 21 5 17]);
a=A(:,[3 19 11 7 10 13 15 1 21 5 17]);
disp(size(a));
% save('sita1.mat','Z1')
% save('sita6.mat','Z2')
% save('sita2.mat','Z3')
% save('sita3.mat','Z4')
% save('sita4.mat','Z5')
% save('sita5.mat','Z6')
% save('xiangying.mat','a')
xlswrite('xiangying.xlsx',a);
```

这句话`a=A(:,[3 19 11 7 10 13 15 1 21 5 17]);` 相当于读取的11个数是11个点的实部/虚部
顺序为：

| Node number  | 1        | 2   | 3   | 4   | 5    | 6   | 7   | 8   | 9   | 10  | 11  |
| ------------ | -------- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- |
| Point ID     | 237      | 728 | 446 | 381 | 416  | 521 | 556 | 118 | 827 | 255 | 587 |
| Point number | 2 (默认实部) | 10  | 6   | 4   | 5 虚部 | 7   | 8   | 1   | 11  | 3   | 9   |
| Line order   | 3        | 19  | 11  | 7   | 10   | 13  | 15  | 1   | 21  | 5   | 17  |

**解决**：
- [ ] 要么重新修改训练数据集(之后的论文要保证FRFs的一致性)
- [x] 要么修改代码 --> 这个问题

```python
if need_select_node:
    print("Select node")
    print(test_FRdata.shape)
    new_test_FRdata[:, :, :, 0] = test_FRdata[:, :, :, 2-1].real
    new_test_FRdata[:, :, :, 1] = test_FRdata[:, :, :, 10-1].real
    new_test_FRdata[:, :, :, 2] = test_FRdata[:, :, :, 6-1].real
    new_test_FRdata[:, :, :, 3] = test_FRdata[:, :, :, 4-1].real
    new_test_FRdata[:, :, :, 4] = test_FRdata[:, :, :, 5-1].imag
    new_test_FRdata[:, :, :, 5] = test_FRdata[:, :, :, 7-1].real
    new_test_FRdata[:, :, :, 6] = test_FRdata[:, :, :, 8-1].real
    new_test_FRdata[:, :, :, 7] = test_FRdata[:, :, :, 1-1].real
    new_test_FRdata[:, :, :, 8] = test_FRdata[:, :, :, 11-1].real
    new_test_FRdata[:, :, :, 9] = test_FRdata[:, :, :, 3-1].real
    new_test_FRdata[:, :, :, 10] = test_FRdata[:, :, :, 9-1].real
    test_FRdatas.append(new_test_FRdata)
```

## Folding Fin
