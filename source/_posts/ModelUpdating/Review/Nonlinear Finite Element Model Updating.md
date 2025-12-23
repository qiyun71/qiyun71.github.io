---
title: Nonlinear Finite Element Model Updating
date: 2025-07-11 22:24:25
tags:
  - 
categories: Blog&Book&Paper/Write/Write Paper/Model Updating
---

| Title       | Nonlinear Finite Element Model Updating |
| ----------- | ------------------- |
| Author      |                     |
| Conf/Jour   |                     |
| Year        |                     |
| Project     |                     |
| Paper       |                     |

<!-- more -->

# Review

> “非线性有限元模型修正发展现状” ([pdf](zotero://open-pdf/library/items/PWXB8PJ9?page=16&annotation=SEPG9TUB))

“基于频响函数数据的局部非线性结构有限元模型修正的一般流程” ([王兴, 2024, p. 681](zotero://select/library/items/HELC7JRX)) ([pdf](zotero://open-pdf/library/items/QEQQ7DYZ?page=14&annotation=8ZF6AZGT))

“利用基础线性模态的分布式非线性结构有限元模型修正流程” ([王兴, 2024, p. 686](zotero://select/library/items/HELC7JRX)) ([pdf](zotero://open-pdf/library/items/QEQQ7DYZ?page=19&annotation=7HPLPEJP))


# 非线性的表现

非线性行为：
- 材料非线性
  - “材料非线性分析主要计算考虑材料应力-应变 关系非线性时的结构响应” ([李钢和余丁浩, 2023, p. 2](zotero://select/library/items/ZD6FLYGM)) ([pdf](zotero://open-pdf/library/items/J5FE6I2U?page=3&annotation=VVGJWPSL))“主要基于固体力学建立的各种类型材料本 构模型,包括弹塑性力学、损伤力学、断裂力学等” ([李钢和余丁浩, 2023, p. 2](zotero://select/library/items/ZD6FLYGM)) ([pdf](zotero://open-pdf/library/items/J5FE6I2U?page=3&annotation=KWUTTV58))
  - 应力-应变响应非线性。最简单的形式是非线性弹性，即应力-应变之间不呈线性关系，更一般的情况是材料对加载和卸载的响应各不相同的情况。 ——庄茁《非线性有限元》
- 几何非线性
  - “几何非线性分析主要计算结构几何形态改变 对其受力变形关系的影响” ([李钢和余丁浩, 2023, p. 3](zotero://select/library/items/ZD6FLYGM)) ([pdf](zotero://open-pdf/library/items/J5FE6I2U?page=4&annotation=H36T6A58))
  - 模型的位移大小影响结构的响应。主要原因有：大挠度/大转动、“突然翻转”、初应力或载荷刚性化等。 ——庄茁《非线性有限元》
- 接触非线性
  - “接触非线性主要模拟因摩擦、碰撞等导致的 边界条件改变” ([李钢和余丁浩, 2023, p. 3](zotero://select/library/items/ZD6FLYGM)) ([pdf](zotero://open-pdf/library/items/J5FE6I2U?page=4&annotation=GMK64TLJ))“接触力学法适 用性广,可用于一般接触非线性问题的计算求解,” ([李钢和余丁浩, 2023, p. 3](zotero://select/library/items/ZD6FLYGM)) ([pdf](zotero://open-pdf/library/items/J5FE6I2U?page=4&annotation=ZVGRKNKR))“接触 单元法通过在相互接触物体间建立特殊单元来模 拟接触非线性,其本质是将接触非线性问题等效 转化为某种特殊类型的材料非线性问题,程序实 现简便,主要适用于相互接触物体相对位移不大 的情况。” ([李钢和余丁浩, 2023, p. 3](zotero://select/library/items/ZD6FLYGM)) ([pdf](zotero://open-pdf/library/items/J5FE6I2U?page=4&annotation=QGMKBDWT))
  - 边界条件在分析过程中发生变化。——庄茁《非线性有限元》

非线性振动现象：
- 刚度弱非线性响应：结构的共振峰会产生“频率漂移”的现象.
- 阻尼弱非线性响应：频响函数的共振峰幅值发生明显变化或自由振动的对数衰减率发生变化.
- 强非线性响应

非线性动力学模型
- 局部非线性结构：建模过程中仅需在连接处建立非线性单元
- 分布式非线性结构：几何非线性结构/材料非线性结构

非线性振动试验方法
- 频响函数试验方法：非线性结构的频响函数是与激励幅值相关的, 一般需要开展多个激励量级下的测试才能够全面描述结构的特征.
- 纯模态试验方法
- 自由衰减试验方法

局部非线性结构有限元模型修正：
- 时域辨识方法：力响应面法、时域非线性子空间辨识方法
- 频域辨识方法：线性图法(定性)、定量辨识包括基于随机激励FRF和基于简谐激励FRF

分布式非线性结构有限元模型修正


## 激励幅值——共振频率

***刚度弱非线性响应***

### 负相关

软刚度系统 (Softening Stiffness)

“当非线性刚度构件引起弱非线性响应时, 结构的共振峰会产生“频率漂移”的现象” ([王兴, 2024, p. 670](zotero://select/library/items/HELC7JRX)) ([pdf](zotero://open-pdf/library/items/QEQQ7DYZ?page=3&annotation=DBKAYABC))

“纵向振动模态频率会随着激励幅值的增大而减小” ([王兴, 2024, p. 671](zotero://select/library/items/HELC7JRX)) ([pdf](zotero://open-pdf/library/items/QEQQ7DYZ?page=4&annotation=6VEB64UU)) “Nonlinear Dynamic Behavior in the Cassini Spacecraft Modal Survey” ([pdf](zotero://open-pdf/library/items/IJXJFHS5?page=1&annotation=TG3DZFPM))

“结构**模态频率**随着**激励幅值**增加而减小的现象” ([王兴, 2024, p. 671](zotero://select/library/items/HELC7JRX)) ([pdf](zotero://open-pdf/library/items/QEQQ7DYZ?page=4&annotation=3W25AWCP))

“natural frequencies varying with position and strength of excitation” ([Ewins, 2000, p. 278](zotero://select/library/items/4YTAYTNI)) ([pdf](zotero://open-pdf/library/items/FQ6YLZIG?page=278&annotation=M2SXZHRQ)) “Example of non-linear system response for different excitation levels.” ([Ewins, 2000, p. 279](zotero://select/library/items/4YTAYTNI)) ([pdf](zotero://open-pdf/library/items/FQ6YLZIG?page=279&annotation=MJ37BPPC))“FRF measurements on non-linear analogue system. (a) Sinusoidal excitation; (b) Random excitation; (c) Transient excitation 共振频率随着激振力的增大而减小” ([Ewins, 2000, p. 280](zotero://select/library/items/4YTAYTNI)) ([pdf](zotero://open-pdf/library/items/FQ6YLZIG?page=280&annotation=TZWYBYRF))

### 正相关

硬刚度系统 (Hardening Stiffness)

“帆板的一阶面外弯曲模态和一阶面内弯曲模态的 共振频率会随着振动幅值的减小而降低” ([王兴, 2024, p. 671](zotero://select/library/items/HELC7JRX)) ([pdf](zotero://open-pdf/library/items/QEQQ7DYZ?page=4&annotation=TE6QABEV)) “非线性振动来源于桅杆 的热屈曲和帆板-组合体连接结构的间隙” ([王兴, 2024, p. 671](zotero://select/library/items/HELC7JRX)) ([pdf](zotero://open-pdf/library/items/QEQQ7DYZ?page=4&annotation=X79FSKZK))

“激励量级越大,第一阶频率越高,第 二阶频率的变化较小.” ([何昊南 等, 2019, p. 1486](zotero://select/library/items/MD7DUKSB)) ([pdf](zotero://open-pdf/library/items/9UTL5AVT?page=11&annotation=8MKVQTPM))

“Fig. 7 shows extracted amplitude-frequency results of the two measure points under different levels of excitation. 激振幅值越大，共振频率越大” ([Liu 等, 2022, p. 187](zotero://select/library/items/7Y4MSA5U)) ([pdf](zotero://open-pdf/library/items/YF39DAPI?page=5&annotation=BNBCPJSK))

“The structure is excited at 3 different forcing levels (F 1⁄40.6 N, F1⁄40.7 N, F1⁄40.8 N).The measured FRFs are shown in Fig. 18. 力越大，频率越大” ([Canbaloğlu和Özgüven, 2016, p. 295](zotero://select/library/items/QXSV9SUV)) ([pdf](zotero://open-pdf/library/items/AWPPTS48?page=14&annotation=NSZZNZT5))

## 激励幅值——共振幅值

***阻尼弱非线性响应***
“构件的阻尼非线性往往会伴随着刚度非线性一起产生, 试验中一个直观的表现是频响函数 的共振峰幅值发生明显变化或自由振动的对数衰减率发生变化” ([王兴, 2024, p. 671](zotero://select/library/items/HELC7JRX)) ([pdf](zotero://open-pdf/library/items/QEQQ7DYZ?page=4&annotation=XLSMZVYE))
“在**不同激励力**下的频响函数曲线, 观察到 **共振峰**呈现渐软效应以及阻尼比随着**激励幅值**增加而变大的规律” ([王兴, 2024, p. 671](zotero://select/library/items/HELC7JRX)) ([pdf](zotero://open-pdf/library/items/QEQQ7DYZ?page=4&annotation=H3GYX272))

“在相同正压力下, 响应幅值随激振力的增大呈现先快速增长后趋于 稳定的趋势” ([高世民 等, 2025, p. 8](zotero://select/library/items/B2DLRSTW)) ([pdf](zotero://open-pdf/library/items/9EF6AV3E?page=8&annotation=JVF8L23U))

## 接触载荷——共振频率

“不同初始正压力的幅频响应曲线表现出典型的软 特性” ([高世民 等, 2025, p. 7](zotero://select/library/items/B2DLRSTW)) ([pdf](zotero://open-pdf/library/items/9EF6AV3E?page=7&annotation=8ZBVTIL9))
“两个共振频率差异较大,表明随着接 触正压力的增大,接触面发生了滑移/黏滞的变化, 呈现显著的非线性特征,” ([高世民 等, 2025, p. 8](zotero://select/library/items/B2DLRSTW)) ([pdf](zotero://open-pdf/library/items/9EF6AV3E?page=8&annotation=N3PQQIUS))

## 接触载荷——共振幅值

“在相同激振力下,随着初 始正压力的增大,共振幅值呈现先减小后增大的 趋势” ([高世民 等, 2025, p. 8](zotero://select/library/items/B2DLRSTW)) ([pdf](zotero://open-pdf/library/items/9EF6AV3E?page=8&annotation=SEIWS6FD))



## 正反扫频的FRF差异

“对于非线性结构而言, 正扫和 反扫测得的频响函数“跳跃点”是显著不同的, 这一差异也被很多学者用作判断结构是否为非线 性的依据” ([王兴, 2024, p. 676](zotero://select/library/items/HELC7JRX)) ([pdf](zotero://open-pdf/library/items/QEQQ7DYZ?page=9&annotation=PHPLLBCA))

“正、反向扫描均出现了跳跃现象,且正向扫描发生 跳跃的频率比反向扫描高,这是硬刚度的特征.” ([何昊南 等, 2019, p. 1486](zotero://select/library/items/MD7DUKSB)) ([pdf](zotero://open-pdf/library/items/9UTL5AVT?page=11&annotation=HTWRTA8V))

“正扫 与 反 扫 的 峰 值 位 置逐渐分离 （正扫频率比反扫大）” ([王强 等, 2019, p. 6](zotero://select/library/items/ZE34UUSL)) ([pdf](zotero://open-pdf/library/items/T9C3UUBK?page=6&annotation=EEUUYLVQ))

## 滞后非线性

“滞后非线性多用 于描述材料的应力与应变之间或力与速度、位移之 间的滞后关系” ([李韶华和杨绍普, p. 1](zotero://select/library/items/6878MI8M)) ([pdf](zotero://open-pdf/library/items/7DECIHDY?page=1&annotation=NSKY84EY))“而迟滞多用于描述系统的时滞和控 制领域中的滞后现象” ([李韶华和杨绍普, p. 1](zotero://select/library/items/6878MI8M)) ([pdf](zotero://open-pdf/library/items/7DECIHDY?page=1&annotation=KVDEFTVA))

“振动环境下, 连接界面在法向发生接触、分离和碰撞; 在切向发生黏 着、摩擦和滑动” ([王东 等, 2018, p. 45](zotero://select/library/items/8YWDQXCL)) ([pdf](zotero://open-pdf/library/items/PZIVMWCG?page=2&annotation=MTNMBEQ4))
“学者们通过试验和理论研究发现连接界面上的 黏滑行为, 表现为恢复力–位移的非线性软化以及迟滞非线性等特征” ([王东 等, 2018, p. 45](zotero://select/library/items/8YWDQXCL)) ([pdf](zotero://open-pdf/library/items/PZIVMWCG?page=2&annotation=N7UGF9ZK))



## 其他

“当非线性项与线 性项比值较大时, 强非线性作用还会引起复杂的振动响应” ([王兴, 2024, p. 673](zotero://select/library/items/HELC7JRX)) ([pdf](zotero://open-pdf/library/items/QEQQ7DYZ?page=6&annotation=FUSXB6AL))
- “正弦扫频试验时发现结构存在明显的多次谐波响应 和亚/超谐波共振等复杂现象” ([王兴, 2024, p. 673](zotero://select/library/items/HELC7JRX)) ([pdf](zotero://open-pdf/library/items/QEQQ7DYZ?page=6&annotation=CTA8TD4T))
- “强非线性系统还会产生混沌等更加复杂的动力学行为” ([王兴, 2024, p. 673](zotero://select/library/items/HELC7JRX)) ([pdf](zotero://open-pdf/library/items/QEQQ7DYZ?page=6&annotation=7NLMKUYL))


