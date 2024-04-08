---
title: An interval model updating strategy using interval response surface models
date: 2024-03-13 13:09:56
tags:
  - 
categories: ModelUpdating/Interval Model Updating
---

| Title     | An interval model updating strategy using interval response surface models                                                                                            |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | S. Fang 福州大学土木工程学院; Qiu-Hu Zhang; W. Ren 、合肥工业大学土木工程学院                                                                                                                |
| Conf/Jour | Mechanical Systems and Signal Processing                                                                                                                              |
| Year      | 2015                                                                                                                                                                  |
| Project   | [An interval model updating strategy using interval response surface models - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0888327015000229) |
| Paper     | https://sci-hub.ee/https://doi.org/10.1016/j.ymssp.2015.01.016                                                                                                        |

<!-- more -->

随机模型更新为处理现实世界结构中存在的不确定性提供了一种有效方法。一般来说，在求解逆问题时会涉及概率论、模糊数学或区间分析。但在实际应用中，由于结构信息不足，往往无法获得结构参数的概率分布或成员函数。此时，区间模型更新程序显示出其在简化问题方面的优越性，因为只需寻求参数和响应的上下限。为此，本研究提出了**区间响应面模型**的新概念，以有效实施区间模型更新程序。这样就能最大限度地避免因使用区间算术而经常出现的区间高估，从而准确估计参数区间。同时，区间反问题的建立被高度简化，并节省了计算成本。通过这种方法，可以实现相对简单和经济高效的区间更新过程。最后，所开发方法的可行性和可靠性已通过数值质量弹簧系统和一组实验测试钢板进行了验证。


# Case

## A numerical mass–spring system


## Experimental validation (Steel Plate Structures)

### Interval estimation of thicknesses

$$\begin{aligned}&f_1^l=63.833-0.027081\left(t_1^l-2.392\right)^2-0.048409\left(t_2^l-20.270\right)^2-0.027081\left(t_3^l-2.392\right)^2,\\&f_2^l=144.432-0.12450\left(t_1^l-8.455\right)^2-0.16342\left(t_2^l-5.954\right)^2-0.12450\left(t_3^l-8.455\right)^2,\\&f_3^l=209.514-0.07554(t_1^l-3.228)^2-0.11650(t_2^l-25.148)^2-0.075554\left(t_3^l-3.228\right)^2,\\&f_4^l=2412.405-0.00421429\left(t_1^l-514.631\right)^2+0.050743\left(t_2^l+3.790\right)^2-0.0042429\left(t_3^l-514.631\right)^2,\\&f_5^l=377.586-0.20986\left(t_1^l-15.203\right)^2-0.070894\left(t_2^l-8.759\right)^2-0.20986\left(t_3^l-15.203\right)^2\end{aligned}$$

### Interval estimation of material properties

$$\begin{gathered}
f_{1}^{l} =77.624-0.0202366\left(E^l-40.905\right)^2+0.010914\left(G^l-2.342\right)^2, \\
f_{2}^{l} =-2214.965+0.002557626\left(E^l+955.633\right)^2+0.099410\left(G^l-2.274)^2\right. \\
f_{3}^{l} =218.882-0.00043976{\left(E^l-181.515\right)}^2-0.084119{\left(G^l-28.413\right)}^2, \\
f_4^l=45.397+0.137681\left(E^l+37.303\right)^2+0.35152\left(G^l-2.201\right)^2,\\
f_5^l=1887.265-0.000257232\left(E^l-2387.009\right)^2-0.139010\left(G^l-31.986\right)^2
\end{gathered}
$$
(GPa)
初始：E$[190, 220]$和G$[77, 89]$
更新后：$E[196.5,203.6]$和$G[79.5,83.4]$

### Interval estimation of geometric and material properties

$$\begin{gathered}
f_{1}^{l} =-158316.423+10^{-5}\times0.75761676(t^l+144591.51)^2-0.019941\left(E^l-40.8297\right)^2+0.010801\left(G^l-2.3428\right)^2, \\
f_2^l =1050593.341-10^{-5}\times0.87364756{\left(t^{\prime}-347169.71\right)}^{2}+0.002308462{\left(E^{\prime}+1041.6762\right)}^{2}+0.098439{\left(G^{\prime}-2.2759\right)}^{2}, \\
f_3^l =-193600.754+10^{-4}\times0.600271859\left(t^{\prime}+56822.63\right)^2-0.00041754\left(E^{\prime}-188.0844\right)^2-0.082790\left(G^{\prime}-28.3916\right)^2, \\
f_4^l =506639.779+10^{-4}\times0.70446129{\left(t^l+84808.55\right)}^2+0.134659{\left(E^l+37.5399\right)}^2+0.34845{\left(G^l-2.2045\right)}^2, \\
 _5^4=327076.625+10^{-3}\times0.148232086\left(t^l+47106.34\right)^2-0.000254016\left(E^l-2378.33\right)^2-0.136660\left(G^l-31.9955\right)^2 
\end{gathered}$$

