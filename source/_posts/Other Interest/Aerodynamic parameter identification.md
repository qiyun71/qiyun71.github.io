---
title: Aerodynamic parameter identification
date: 2026-02-03 16:09:27
tags:
categories: Other Interest
Year:
Journal:
---

> [Gemini 3 Pro](https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221xPmlRKhQg-mE30j239z0uHzSK8_n8PtK%22%5D,%22action%22:%22open%22,%22userId%22:%22107403435292964343607%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing)

气动参数辨识目标是根据飞行器的输入（飞行参数）输出（速度）数据，辨识出飞行器气动特性参数的数值，从而建立飞行器气动模型，为飞行器的设计、控制和性能评估提供依据。

气动参数（气动力/力矩系数）是描述飞行器在飞行过程中所受气动力和力矩与飞行状态及控制输入之间关系的参数，通常包括升力系数、阻力系数、侧力系数、滚转力矩系数、俯仰力矩系数和偏航力矩系数等。辨识完成后可以构成飞行器的数学模型，计算效率远高于CFD仿真，适**用于飞行器的实时控制和仿真**。
- 飞控需要知道气动参数，以确定每个操纵面的控制效果，从而实现对飞行器的稳定和控制。
- 模拟器需要气动参数来准确模拟飞行器的飞行性能和响应特性。
- 故障诊断和健康监测也需要气动参数来评估飞行器的状态和性能。

在地面通过风洞/CFD辨识完成的气动参数，往往不能完全反映飞行器在实际飞行环境中的气动特性，因此需要通过飞行试验数据进行气动参数辨识，以提高气动模型的准确性和可靠性。

通常把气动参数分为**纵向**（抬头/低头、加减速）和**横航向**（滚转、偏航）两组，分别决定飞机能否平稳飞行和能否转弯走直线，可以通过这些数据辨识：运动状态响应（速度、加速度）和气动状态数据（马赫数、攻角、侧滑角）和激励（舵面偏转角、发动机推力）。**响应=气动参数x气动状态/激励**

<!-- more -->

## 气动模型

> “气动力建模一般形式” ([王超 等, 2019, p. 489](zotero://select/library/items/IQ4TU6M8)) ([pdf](zotero://open-pdf/library/items/9CD4LKZD?page=2&annotation=ZYQ48MF5))

气动力模型是表征飞行器的气动力/力矩与飞行状态、控制输入等参数之间关系的函数。通常可表做一般形式：
$C_i = f(\alpha, \beta, Ma, \delta_j, ...)$
其中，$C_i$表示某一气动力或力矩系数，$\alpha$为攻角，$\beta$为侧滑角，$Ma$为马赫数，$\delta_j$为第j个控制面偏转角，...表示其他可能影响气动力的参数。

## Code

[peterdsharpe/NeuralFoil: NeuralFoil is a practical airfoil aerodynamics analysis tool using physics-informed machine learning, exposed to end-users in pure Python/NumPy.](https://github.com/peterdsharpe/NeuralFoil)


## Identification methods


> “获取飞行器气动力参数的方法主要 有数值模拟、风洞试验以及飞行试验 3 种” ([崔瀚 等, 2025, p. 14](zotero://select/library/items/M2PEG69M)) ([pdf](zotero://open-pdf/library/items/P8LUWCR3?page=2&annotation=4FEESZ3X))

- 数值模拟（CFD）基于计算流体力学,依赖数学模型和理想性假设,存在**计算方法和模型误差以及计算误差**
- 风洞实验依据相似性原理,通过模型测试获取数据,获取气动力参数的**成本较高**且精度受限于流场品质、传感器及数据采集系统精度以及洞壁和支架干扰的修正
- 飞行试验(Quick Access Recorder,  QAR)在实际环境中进行,能够真实反映气动特性,但气动力参数需通过系统辨识理论间接获得,故辨识方法的选择成为了重要环节

### Data-driven

> “当导弹飞行时,纵横向运动相互耦合,必须采用**六自由度运动方程**作为导弹动力学系统的数学模型,其中气动参数是未知的” ([浦甲伦 等, 2018, p. 2](zotero://select/library/items/VLFM5XE8)) ([pdf](zotero://open-pdf/library/items/WLND75IM?page=2&annotation=AW56DW4C))

### PINNs

> “飞机纵向运动” ([付军泉 等, 2023, p. 32](zotero://select/library/items/4U2MQWCN)) ([pdf](zotero://open-pdf/library/items/ZJJF7C74?page=3&annotation=HC47FUQU))
> “The longitudinal state equation” ([Cook, 2007, p. 138](zotero://select/library/items/BL9IHMVS)) ([pdf](zotero://open-pdf/library/items/6U47H9CJ?page=161&annotation=I287I6UA))

通过数据MAE和ODE损失训练PINN：
- 输入为流向速度$u$、垂向速度$w$、俯仰角速率$q$、俯仰角$\theta$和升降舵偏转角$\delta_{e}$
- 输出为下一时刻的流向速度$u$、垂向速度$w$、俯仰角速率$q$。

状态方程中的多个待辨识的气动参数

> “飞行器六自由度非线性模型” ([刘磊 等, 2025, p. 3](zotero://select/library/items/BJ7XAP83)) ([pdf](zotero://open-pdf/library/items/TBJN79T5?page=3&annotation=3NH9RTR2))
> 高超声速飞行器( hypersonic vehicle, HSV)

气动系数方程（滚转$C_{l}$, 俯仰$C_{m}$, 偏航力矩系数$C_{n}$）中的多个待辨识参数

PINN
- 输入为p、q、r  分别表示滚转、俯仰、偏航角速度，V、α、 β 分别表示飞行器速 度,攻角,侧滑角，da、  de、dr 分别表示飞行器的左副翼、右副翼与方向舵的舵偏量，Ma 表示飞行器马赫数
- 输出为预测的滚转、俯仰、偏航角的角加速度$\dot{p},\dot{q},\dot{r}$


> “体轴系下六分量气动力系数的数学模型” ([李耀, 2025, p. 38](zotero://select/library/items/T2TGGP2X)) ([pdf](zotero://open-pdf/library/items/HA2U2YWS?page=38&annotation=7EZZN695))
> [司海青](https://cca.nuaa.edu.cn/_t447/2017/1110/c4797a102638/page.htm) 气动参数辨识、不确定性

将QAR数据辨识的结果作为试验数据：
- 高保真数模用激光扫描重建
- QAR数据记录飞行参数，列举了10个对气动力和气动力矩影响较大的参数作为输入特征
- 通过数学模型计算气动力和气动力矩作为输出标签

对CFD的结果进行验证和确认

考虑不确定性：统计一段时间内的攻角和速度分布来量化？


