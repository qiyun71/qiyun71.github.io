---
title: Multiphysics
date: 2025-11-29 20:22:34
tags: 
categories: Learn/Finite Element
Year: 
Journal:
---

多物理场耦合(Multiphysics Coupling)是指在有限元分析中同时考虑多个物理现象之间的相互作用和影响。这些物理现象可以包括结构力学、热传导、流体力学、电磁学等。通过多物理场耦合，可以更准确地模拟和分析复杂系统的行为，捕捉不同物理场之间的相互作用，从而提高仿真结果的精度和可靠性。

<!-- more -->

# Talk

[高超声速飞行器复杂流固热耦合问题研究——叶坤_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1sS4y1f7Gi/?spm_id_from=333.337.search-card.all.click&vd_source=1dba7493016a36a32b27a14ed2891088)
[叶坤的个人主页-西北工业大学](https://teacher.nwpu.edu.cn/yekun.html)
流固热耦合：
- 气动热难以预测：工程方法无法用于复杂外形，CFD计算量大
  - 工程算法忽略很多因素，采用参考温度和参考焓法计算热流，采用活塞流理论计算气动力
  - CFD求解NS方程，计算量大。受网格、湍流模型、计算格式以及真实气体效应等因素影响
  - 风洞试验成本高，且存在尺度效应
- 时间尺度跨度大(时间步长)：流动时间1e-4~1e-5，振动时间1e-2~1e-3，传热时间1~10
  - 流动（CFD的时间步长）：气体从机头传到机尾的时间为L/v，L为飞行器长度，v为飞行速度。这是远远不够的，需要更小的时间步长来捕捉流动的细节，一般$\nabla T=\frac{L}{vn}$，让气体从机头传到机尾的时间经历过n个时间步长
  - 结构（CSD的时间步长）：结构的时间步长受最高频率影响，一般为$f_{max}$对应的周期的1/10，即$\Delta t_{structure}=\frac{1}{10f_{max}}$
  - 传热（CTD的时间步长）：传热的时间步长受热扩散影响，一般为$\Delta t_{thermal}=\frac{\rho c \Delta x^2}{2k}$，$\Delta x$为结构单元的最小尺寸
- 物理机理复杂：高超声速气动热、非线性结构动力学、材料高温性能退化等


# Basic

## 气动热力学

气动热 <==> 气动力

[(36 封私信 / 6 条消息) DIMAXER 流热耦合 | 高超声速飞行器极端多物理场高精度仿真 - 知乎](https://zhuanlan.zhihu.com/p/1943252235422262710)

气动力计算方法“主要分为工程计算方法和 CFD 数值计算方法两大类” ([桂业伟 等, 2017, p. 3](zotero://select/library/items/84LD63YT)) ([pdf](zotero://open-pdf/library/items/MRRKLXDY?page=3&annotation=7MTRHV5D))

气动热计算方法([桂业伟 等, 2017, p. 3](zotero://select/library/items/84LD63YT)) ([pdf](zotero://open-pdf/library/items/MRRKLXDY?page=3&annotation=T25RS9RB))
- 纯粹的数值方法,直接求解 Navier-Stokes方程及其近似形式
- 完全的工程方法
- 边界层外无黏数值求解与边界层内工程估算结合方法 

> [(39 封私信) 湍流的描述方程LES-NS（大涡模拟）和RANS（雷诺平均）有什么区别？ - 知乎](https://www.zhihu.com/question/20886265/answer/36169331)

LES中文名大涡模拟，基本思想是对NS方程进行某种过滤，然后只计算大尺度的湍流，而将小于过滤尺度的湍流用模型加以刻画。数学上，小于过滤尺度的湍流表现为额外的应力项，称为亚网格应力。现有的湍流理论已经有结论，几乎所有的湍流在足够小的尺度上都具有一定的相似性。也就是说，用一个普适的模型来近似亚网格应力在理论上是可能做到的，虽然目前还没有出现这样的模型。

RANS中文名雷诺平均NS方程，基本思想是对NS方程进行（时间）平均，将非定常的湍流问题转化为一个定常的问题研究，代价是会出现额外的未知数，形式上也和应力的地位相同，称为雷诺应力。雷诺应力同样需要模型进行刻画，这也就是所谓的湍流模式或者湍流模型。然而，由于对问题进行了（时间）平均，方程本身包含的信息已经部分丢失，给出雷诺应力的模型实际上非常困难，同时也很难做到对所有流动都适用。

“Table 1: Simulation_techniques.” ([Chowdhury, 2024, p. 48](zotero://select/library/items/6382AVJG)) ([pdf](zotero://open-pdf/library/items/EI2CPJMH?page=4&annotation=6ANZ8MIK))

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20260128104916.png)


## 热-结构动力学

传热：热传导、热对流、热辐射

[(30 封私信 / 2 条消息) 详解热应力：从原理到应用，掌握控制与管理的那些事儿 - 知乎](https://zhuanlan.zhihu.com/p/30546924957)

温度变化引起材料热膨胀或收缩，若这种变形受到约束，就会在材料内部产生应力，这种应力称为热应力。

对于线性弹性材料，热应力$\sigma=E\alpha \Delta T$ (Pa)
- $E$：材料的弹性模量(Pa)
- $\alpha$：材料的线膨胀系数(1/°C)
- $\Delta T$：温度变化(°C)


## 基础概念

[一文掌握Abaqus多物理场耦合_多场耦合-CSDN博客](https://blog.csdn.net/Cassplm/article/details/148845357)

“多 场 、多 域 和 多 尺 度 的 概 念” ([桂业伟 等, 2017, p. 2](zotero://select/library/items/84LD63YT)) ([pdf](zotero://open-pdf/library/items/MRRKLXDY?page=2&annotation=PIW5M6CN))
- 多场是指分析中的物理场,包括气动场、热场、结构场等
- 多域是指存在共同的边界条件且存在相互作用的系统 ,包括气动域和结构域

## 耦合影响因素

> 飞行器表现取决于推进力与空气动力的平衡，当推进力为0(滑行)时，飞行器的运动完全由空气动力决定

“These loads will depend on the **flight condition** (altitude, Mach number and attitude) as well as the **shape or the outer mould line** (OML) of the vehicle, and all can be influenced, at least in part, by FTSI (Figure 2). **Higher velocities, lower altitudes and higher inclination angles** will all increase both the rate of convective heat transfer to the vehicle and the aerodynamic pressure imposed on the structure.” ([Neely 等, 2025, p. 3](zotero://select/library/items/LPJ25XZ3)) ([pdf](zotero://open-pdf/library/items/MLRT8ECW?page=3&annotation=NGYCNKUY))

# 气动-热-结构(流-热-固) 

Aerodynamic-thermal-structural

1958 年 , Roger .M Aerothermoelasticity

> 没有考虑阻尼吗？

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251201161928.png)

(1) 热量产生和弹性变形之间的热力学耦合 可以忽略不计; 
(2) 动气动弹性弱耦合,气动弹性特征时间远小于气动热/结构传热特征时间;
(3) 静气动弹性弱耦合,由于稳态压力和热载荷导致的静态弹性变形不足以改变结构温度分布。

⭐⭐⭐“耦合问题分析方法” ([李莎靓 等, 2025, p. 22](zotero://select/library/items/TW87Z6LQ)) ([pdf](zotero://open-pdf/library/items/SULM2UMP?page=3&annotation=QELY7EEU))

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251202114357.png)

“Lohner等[105]首次提出了紧 耦 合 和 松 耦 合 的 概念” ([桂业伟 等, 2017, p. 8](zotero://select/library/items/84LD63YT)) ([pdf](zotero://open-pdf/library/items/MRRKLXDY?page=8&annotation=HZ7WE2RG))
- 紧耦合：耦合时间步长采用流场的特征时间，二阶时间精度，仿真计算量大
- 松耦合：耦合时间步长采用结构热响应的特征时间，一阶时间精度

“耦合方法的应用” ([李莎靓 等, 2025, p. 32](zotero://select/library/items/TW87Z6LQ)) ([pdf](zotero://open-pdf/library/items/SULM2UMP?page=13&annotation=5NFB7CMM))
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20260113142324.png)


“**气动热力耦合关系** from 中国空气动力研究与发展中心” ([桂业伟 等, 2017, p. 3](zotero://select/library/items/84LD63YT)) ([pdf](zotero://open-pdf/library/items/MRRKLXDY?page=3&annotation=3MUBMLF5)) 

“桂业伟 [44-45] 等 自主研发的热环境/热响应耦合计算分析平台 FLCAPTER (Coupled Analysis Platform for Thermal Environment and Structure Response),基于热环境、 热防护、热管理、热布局等研究基础,耦合分析 高速飞行器气动热与热防护综合设计问题。” ([李莎靓 等, 2025, p. 25](zotero://select/library/items/TW87Z6LQ)) ([pdf](zotero://open-pdf/library/items/SULM2UMP?page=6&annotation=XL78N5I6))

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251130212758.png)

根据研究者的能力和理解水平,可将其分为3个不同耦合问题:
- 热-固耦合问题：其主要解决的是流-固交界面上气动热与结构表面温度间的双向强耦合问题。
- 流-固耦合问题：其主要解决的是流-固耦合界面上气动力、惯性力与弹性力间的耦合问题
- 流-热-固耦合问题：其主要解决的是流-固交界面上气动力、惯性力、弹性力与气动热、结构温度间的多物理场强耦合问题。

“Basic structure of the aerothermoelastic problem” ([McNamara和Friedmann, 2011, p. 1091](zotero://select/library/items/B3DV68PC)) ([pdf](zotero://open-pdf/library/items/8EE7FMJW?page=3&annotation=MA3LAIM8))

考虑②(结构变形对气动热的反馈)即为双向流-热-固耦合

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20260112194616.png)

图中: ([桂业伟 等, 2017, p. 6](zotero://select/library/items/84LD63YT)) ([pdf](zotero://open-pdf/library/items/MRRKLXDY?page=6&annotation=RWF89DM7))
- $T_{w}$ 为壁面温度
- $h_{c}$ 为对流传热系数
- $T_{aw}$ 为绝热壁温度
- $p_{e}、T_{e}、Ma_{e}$ 分别为边界层边缘的压力、温度、 马赫数
- $T_{struct}$为结构温度
- $w$和$\dot w$ 分别为横向位移和速度
- $P_{surf}$为表面压力”

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20260113123609.png)

“多 场 耦 合 问 题 场 间 数 据 流 7K 意 图” ([桂业伟 等, 2015, p. 3](zotero://select/library/items/KWHRAS4J)) ([pdf](zotero://open-pdf/library/items/BCS82ZJ5?page=3&annotation=HFEMDQBN))
![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251201161804.png)

“Physical coupling within fluid-thermal-structural interaction (FTSI) with respect to high-speed vehicles.” ([Neely 等, 2025, p. 2](zotero://select/library/items/LPJ25XZ3)) ([pdf](zotero://open-pdf/library/items/MLRT8ECW?page=2&annotation=BN6G8WWS))

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20260113121811.png)

“Quasi-static solution procedure.” ([Culler和McNamara, 2011, p. 2399](zotero://select/library/items/3T3HDNMV)) ([pdf](zotero://open-pdf/library/items/YN64TZBW?page=7&annotation=8566NCTP))

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20260113150618.png)


## Equation

“数值计算理论” ([张佳明 等, 2021, p. 136](zotero://select/library/items/IXV8JCN9)) ([pdf](zotero://open-pdf/library/items/AYFS4YAW?page=2&annotation=2SJ8BGXV))


| Examples                                                                                                                                                                                                                          | Methods                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| “数值模拟” ([张佳明 等, 2021, p. 88](zotero://select/library/items/5AAZBJMU)) ([pdf](zotero://open-pdf/library/items/3MBBUGIG?page=2&annotation=BWUV599F))<br><br>                                                                        | ![image.png\|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251201152424.png)<br>同时建立流体域模型和固体域模型,利用基于有限体积法的**Fluent求解器**,通过求解连续方程、动量守恒方程和能量守恒方程,计算流体域的温度、压强、速度及耦合面上的温度分布。然后通过**SystemCoupling模块将流体域网格节点上的热流和压力数据插值映射到固体域表面网格上**,并作为结构场求解的边界条件。利用**Transient Structural求解器**计算结构上的温度、应力、应变与位移分布。然后再此通过**SystemCoupling模块将结构的温度和位移场数据插值映射到流体域网格上**,并以此作为边界条件进行流体域的求解,直到达到所需的耦合计算时间。                                                                                                                                                                                                                                                                                                                                      |
| “耦合计算策略” ([刘深深 等, 2025, p. 120](zotero://select/library/items/MNY8A8JD)) ([pdf](zotero://open-pdf/library/items/A7Y33S6Z?page=3&annotation=KRNRBEEC))                                                                             | ![image.png\|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251201160854.png)<br>1. 根据结构几何形态和热力学状态，计算气动力载荷和热流分布<br>2. 将气动压力和热流载荷传递至结构场，计算结构变形和温度场分布<br>3. 将结构变形后的几何外形与温度场变化反馈至气动力热数值求解器                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| “Fig. 1. Schematic of fluid-thermal-structural analysis framework based on CFD/CTSD.” ([Yang 等, 2021, p. 3](zotero://select/library/items/94373JVT)) ([pdf](zotero://open-pdf/library/items/2LWPPHJY?page=3&annotation=CWVDHSUG)) | ![image.png\|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251202142816.png)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| [(34 封私信 / 20 条消息) ANSYS Workbench 中如何使 Transient Structure 模块实现多核求解？ - 知乎](https://www.zhihu.com/question/59049404)                                                                                                              | 我现在使用ANSYS Workbench + Fluent + Transient Structure + System Coupling 的组合实现了一个 2-way Fluid-Solid Interaction 瞬态模型。 此模型描述的是固体颗粒在流体中“随波逐流”的过程。  <br>我现在遇到了一个问题：  <br>- Fluent模块可以完全利用CPU提供的八个线程进行并行计算，从而获得不错的加速效果。  <br>- 但是Transient Structure模块使用的是 Mechanical APDL 的求解器，只能使用两个线程，导致仿真的瓶颈居然出现在固体力学方程求解这一部分  <br>每次使用workbench开始仿真的时候，都会出现一个名为 “Distributed Mechanical APDL with requesting 2 cores" 的窗口，可以由此得出结论，Mechanical模块只利用了两个线程。  <br>试过以下的方法：  <br>1. 设置license的使用方式为 use a separate license for each application, 然后在options -> Solution Process 中设置 Default Execution mode 为 Parallel  <br>2. 设置 options ->Mechanical APDL 中设置 Processors = 8； 设置Command Line options = -np 8  <br>以上方法均无效。  <br>向知乎上各位大神求教，在下一介苦逼博士生，希望用ANSYS仿真以完成科研任务，先在此谢谢各位了！ |

| Fluid-Solid                                                                                                                                                        |                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------- |
| [【ANSYS流固耦合】基本流程_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1ef4y1S7se/?spm_id_from=333.337.search-card.all.click&vd_source=1dba7493016a36a32b27a14ed2891088) | Fluent+System Coupling+Transient Structural |

## Case

### Simulation

[Software · MDO Lab](https://mdolab.engin.umich.edu/software)

| Case                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| “圆管绕流试验” ([张佳明 等, 2021, p. 137](zotero://select/library/items/IXV8JCN9)) ([pdf](zotero://open-pdf/library/items/AYFS4YAW?page=3&annotation=AHEPU8NU)) Allan R. Wieting 在 NASA LANGLEY 8 - foot 高温风洞                                                                                                                 | ![image.png\|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251201150822.png)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| “舵翼结构模型” ([张佳明 等, 2021, p. 138](zotero://select/library/items/IXV8JCN9)) ([pdf](zotero://open-pdf/library/items/AYFS4YAW?page=4&annotation=5HEGZQBV))<br>“舵翼结构气动热力耦合模拟” ([张佳明 等, 2021, p. 89](zotero://select/library/items/5AAZBJMU)) ([pdf](zotero://open-pdf/library/items/3MBBUGIG?page=3&annotation=E4YQX434)) | ![image.png\|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251201151504.png)<br>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|                                                                                                                                                                                                                                                                                                                     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |


| Aerodynamic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |                                                                                                                                                                 |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [WyllDuck/OpenFOAM-ToolChain-for-Rocket-Aerodynamic-Analysis: This project provides a complete toolchain for evaluating different rocket geometries at subsonic, transonic, and supersonic regimes. The work contained in this repository is part of a student project carried out at the Technical University of Munich (TUM) under the Master of Science (M.Sc) in Aerospace (year 2023).](https://github.com/WyllDuck/OpenFOAM-ToolChain-for-Rocket-Aerodynamic-Analysis)<br>[WyllDuck/OpenFOAM-ToolChain-helperFunctions: WARR thermal simulation](https://github.com/WyllDuck/OpenFOAM-ToolChain-helperFunctions?tab=readme-ov-file) | ![results.png (898×865)\|333](https://raw.githubusercontent.com/WyllDuck/OpenFOAM-ToolChain-for-Rocket-Aerodynamic-Analysis/refs/heads/main/images/results.png) |
| [web.mit.edu/drela/Public/web/xfoil/](https://web.mit.edu/drela/Public/web/xfoil/)<br>“aerodynamic analysis tool XFOIL” ([Wang 等, p. 3](zotero://select/library/items/LY9APXAJ)) ([pdf](zotero://open-pdf/library/items/AFFVKZHD?page=3&annotation=2CMV37HL))                                                                                                                                                                                                                                                                                                                                                                             | ![xfoil_logo.gif (503×118)](https://web.mit.edu/drela/Public/web/xfoil/xfoil_logo.gif)                                                                          |






### Experiment

“集束射流气动热环境模拟实验原理示意图” ([张佳明 等, 2021, p. 88](zotero://select/library/items/5AAZBJMU)) ([pdf](zotero://open-pdf/library/items/3MBBUGIG?page=2&annotation=YG52KZVN))

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251201152655.png)


Measurement：
- 气体压力：喷管出口处流场参数。“测量探头使流场产生激波,测量得到的压力为激波后的 流场总压,流场实际静压值为:” ([张佳明 等, 2021, p. 92](zotero://select/library/items/5AAZBJMU)) ([pdf](zotero://open-pdf/library/items/3MBBUGIG?page=6&annotation=U8EZVRZ5))
- 结构变形：高温应变片布置在舵翼结构尾部背风面上部
- 结构温度：“将 S 型热电偶布置在背风面下部” ([张佳明 等, 2021, p. 92](zotero://select/library/items/5AAZBJMU)) ([pdf](zotero://open-pdf/library/items/3MBBUGIG?page=6&annotation=D8RCGLCA))