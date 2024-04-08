---
title: Interval Identification of Structural Parameters Using Interval Deviation Degree and Monte Carlo Simulation
date: 2024-03-12 16:48:16
tags:
  - 
categories: ModelUpdating/Interval Model Updating
---

| Title     | Interval Identification of Structural Parameters Using Interval Deviation Degree and Monte Carlo Simulation                                                                                                 |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Zhaopu Guo and Zhongmin Deng∗ 北航                                                                                                                                                                            |
| Conf/Jour | International Journal of Computational Methods                                                                                                                                                              |
| Year      | 2019                                                                                                                                                                                                        |
| Project   | https://www.worldscientific.com/doi/abs/10.1142/S0219876218501037                                                                                                                                           |
| Paper     | [Interval Identification of Structural Parameters Using Interval Deviation Degree and Monte Carlo Simulation](https://readpaper.com/pdf-annotate/note?pdfId=2202709185245307648&noteId=2202709381471330048) |

<!-- more -->

# AIR

本文从不确定性传播和不确定性量化两个方面研究了结构参数的区间辨识问题。将蒙特卡罗(MC)模拟与代理模型相结合，可以有效地获得结构响应的精确区间估计。利用区间长度的概念，**构造了区间偏差度(IDD)来表征解析模态数据与实测模态数据间区间分布的不一致**。通过求解两个优化问题，很好地估计了系统参数的标称值和区间半径。最后，通过数值和实验验证了该方法在结构参数区间识别中的可行性。

算例：
- Numerical Case Studies: **A Mass-Spring System**
- Experimental Case Study: **Steel Plate Structures**
- Complex Case Study: Satellite Structure


# Method

一种新的UQ指标：interval deviation degree (IDD)
MC进行不确定性传播

# Case

## Numerical Case Studies: A Mass-Spring System


### Case 1: Well-separated modes


$$\begin{aligned}
m_{1}& =1(\mathrm{kg}),\quad m_2=1(\mathrm{kg}),\quad m_3=1(\mathrm{kg});  \\
k_{3}& =k_4=1\mathrm{(N/m)},\quad k_6=3\mathrm{(N/m)};  \\
k_{1}& =k_2=k_5=[0.8,1.2]\text{(N/m).} 
\end{aligned}$$

$\begin{aligned}\omega_1^2&=0.2840+0.3416k_1+0.4122k_2+0.0078k_5+0.0745k_1k_2+0.0011k_1k_5\\&-0.0014k_2k_5-0.0423k_1^2-0.0753k_2^2-0.0020k_5^2,\end{aligned}$
$\begin{aligned}\omega_2^2&=1.6117+0.1249k_1+0.5882k_2+1.7402k_5-0.0735k_1k_2+0.1243k_1k_5\\&-0.0015k_2k_5-0.0021k_1^2+0.0748k_2^2-0.1871k_5^2,\end{aligned}$
$\begin{aligned}\omega_3^2&=7.1036+0.5331k_1+0.0001k_2+0.2531k_5-0.0014k_1k_2-0.1247k_1k_5\\&+0.0025k_2k_5+0.0444k_1^2+0.0007k_2^2+0.1885k_5^2,\end{aligned}$
$\begin{aligned}|\varphi(1,1)|&=0.5642-0.0894k_1+0.1060k_2+0.0171k_5+0.0082k_1k_2+0.0059k_1k_5\\&-0.0194k_2k_5+0.0009k_1^2-0.0150k_2^2-0.0012k_5^2.\end{aligned}$

初始$k_{1} = k_{2} = k_{5} = [0.5, 1.5]$
更新后：
- $k_1:[0.80, 1.20]$
- $k_2:[0.80, 1.19]$
- $k_{3}: [0.80, 1.20]$

### Case 2: Close modes

$$\begin{aligned}
m_{1}& =1(\mathrm{kg}),\quad m_2=4(\mathrm{kg}),\quad m_3=1(\mathrm{kg});  \\
k_{1}& =k_3=0\mathrm{(N/m)},\quad k_6=1\mathrm{(N/m)};  \\
k_{2}& =[7.5,8.5](\mathrm{N/m}),\quad k_4=k_5=[1.8,2.2](\mathrm{N/m}). 
\end{aligned}$$

$\begin{aligned}\omega_1^2&=-0.0002+0.0830k_2+0.0839k_4+0.0842k_5+0.0186k_2k_4+0.0185k_2k_5\\&-0.0094k_4k_5-0.0046k_2^2-0.0325k_4^2-0.0325k_5^2,\end{aligned}$
$\begin{aligned}\omega_2^2&=1.6103+0.0104k_2+1.0455k_4-0.0937k_5-0.0097k_2k_4+0.0055k_2k_5\\&+0.0094k_5+0.0042k_2^2+0.0396k_4^2+0.0005k_5^2,\end{aligned}$
$\begin{aligned}\omega_3^2&=1.1103+0.0273k_2+0.0162k_4+1.1572k_5-0.0003k_2k_4-0.0165k_2k_5\\&+0.0104k_4k_5+0.0065k_2^2-0.0034k_4^2+0.0372k_5^2,\end{aligned}$
$\begin{aligned}|\varphi(1,1)|&=0.6658+0.0125k_2-0.0988k_4+0.0496k_5-0.0062k_2k_4+0.0072k_2k_5\\&+0.0020k_4k_5-0.0005k_2^2+0.0190k_4^2-0.0170k_5^2.\end{aligned}$

初始|更新后：
- $k_{2} =[6.5, 9.5]$|$k_{2}=[7.46, 8.52]$
- $k_{4} =[1.6, 2.4]$|$k_{4}=[1.80, 2.20]$
- $k_{5}=[1.5, 2.4]$|$k_{5}=[1.81, 2.20]$

## Experimental Case Study: Steel Plate Structures

- 55块名义上的钢板参数：
  - 600 mm (length)× 120 mm (width) × 3 mm (thickness)
  - Young’s modulus (E) of 210 GPa
  - shear modulus (G) of 83 GPa
  - mass density of 7,860 $kg/m^3$
- 前**五阶频率**来修正钢板的几何和材料参数
- a second-order response surface model 来代替复杂的FE模型

$$\begin{gathered}
f_{1} =13.1+0.2152E-0.01455G-0.0002813E^2-0.0004878E\cdot G+0.0006576G^2, \\
f_{2} =44.08+0.5145E-0.1333G+0.0002943E^2-0.004055E\cdot G+0.005588G^2, \\
f_{3} =52.85-0.1156E+1.445G+0.0006092E^2+0.0002019E\cdot G-0.005634G^2, \\
f_{4} =375.4-3.207E-0.4291G+0.02202E^2-0.009699E\cdot G+0.01383G^2, \\
f_{5} =79.55+0.255E+2.804G-0.0005059E^2-0.001249E\cdot G-0.009833G^2. 
\end{gathered}$$

初始：E$[190, 220]$和G$[77, 89]$
更新后：$E:[196.2, 204.8]$和$G:[79.2, 83.6]$

RSM：[An interval model updating strategy using interval response surface models](An%20interval%20model%20updating%20strategy%20using%20interval%20response%20surface%20models.md)

