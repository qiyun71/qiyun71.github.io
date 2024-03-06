---
title: ReadPaper
date: 2024-02-27 17:06:54
tags:
  - 
categories: ModelUpdating
---

Paper about model updating

<!-- more -->

# Review
## 有限元模型修正研究进展:从线性到非线性

[有限元模型修正研究进展:从线性到非线性](https://readpaper.com/pdf-annotate/note?pdfId=2042966807754524672&noteId=2150699945082996480) 张 皓 李东升 李宏男

- 线性有限元模型修正
  - 传统有限元模型修正方法(基于动力的有限元模型修正仍为本领域研究的主流)，根据修正的对象可分为：
    - 矩阵型方法
    - **参数型方法** 
      - 基于灵敏度的有限元模型修正方法
        - 基于灵敏度的参数法在实际工程中的应用最为广泛,理论研究也相对成熟. 但这类方法的缺点在于灵敏度矩阵的计算, 不仅计算量大, 且容易带来求解问题的病态导致方法失效; 另外, 在应用于大型结构时, 由于迭代运算需要反复调用有限元模型, 造成巨大的计算量, 也限制了其更广泛的应用
      - 采用缩聚模型或代理模型 (surrogate model) 的方法, 以及不基于灵敏度的参数型方法
        - CMCM：不基于灵敏度的参数型方法，交叉模型交叉模态法 (cross-model cross-mode method, CMCM) 不需要进行迭代运算, 且不需要计算灵敏度
        - 神经网络法：BP、CNN
        - 响应面法：代理模型
          - 响应面法得到了广泛关注, 特别是其中所包含的统计思想符合有限元模型确认的发展方向, 在大型结构的有限元模型修正过程中也体现出了修正多个参数的能力。但是, 神经网络法和响应面法的成功依赖于样本空间的选择, 不同的选择产生不同的修正结果, 选择不当也可能会造成代理模型泛化能力不足, 从而导致有限元模型修正结果外推不可靠。增加数据量能够增加模型的可靠性, 但又会带来较大的计算量. 所以, 通过响应面法得到的修正模型必须经过模型确认过程
        - 结合模拟退火、遗传算法、粒子群算法等优化算法的方法
- 非线性有限元模型修正发展现状

**有限元模型确认**是传统有限元模型修正方法在统计理论上的发展, 从理论上具有更一般的意义, 且它能够从理论上探究复杂结构不确定性的传递, 量化评价修正后模型的不确定性, 给出指导工程应用的置信度, 对于修正后有限元模型进一步应用于结构损伤识别、状态评估、性能预测等具有重要实际意义.


## Stochastic Model Updating with Uncertainty Quantification: An Overview and Tutorial

[Stochastic Model Updating with Uncertainty Quantification: An Overview and Tutorial](https://readpaper.com/pdf-annotate/note?pdfId=2201469630320002560&noteId=2201470551774580736)

本文概述了随机模型更新的理论框架，包括模型参数化、敏感性分析、代理建模、试验分析相关性、参数校准等关键方面。**特别关注不确定性分析，将模型更新从确定性域扩展到随机域**。不确定性量化度量极大地促进了这种扩展，不再将模型参数描述为未知但固定的常数，而是具有不确定分布的随机变量，即不精确概率。因此，随机模型更新的目标不再是对单个实验进行最大保真度的单个模型预测，而是将多个实验数据的完全离散性包裹起来，减少模拟的不确定性空间。这种不精确概率的量化需要一个专门的不确定性传播过程来研究如何通过模型将输入的不确定性空间传播到输出的不确定性空间。本教程将详细**介绍前向不确定性传播和反向参数校准这两个关键方面**，以及p盒传播、基于距离的统计度量、马尔可夫链蒙特卡罗采样和贝叶斯更新等关键技术。通过解决2014年**NASA多学科UQ挑战**演示了整体技术框架，目的是鼓励读者在本教程之后复制结果。第二次实际演示是在一个新设计的基准测试台上进行的，在那里制造了**一系列具有不同几何尺寸的实验室规模的飞机模型**，遵循预先定义的概率分布，并根据其固有频率和模型形状进行测试。这样的测量数据库自然不仅包含测量误差，更重要的是包含来自结构几何预定义分布的可控不确定性。最后，讨论了开放性问题，以实现本教程的动机，为研究人员，特别是初学者，提供了从不确定性处理角度更新随机模型的进一步方向。

算例：
- NASA UQ挑战
- 飞机模型

# 随机模型修正

## A robust stochastic model updating method with resampling processing

[A robust stochastic model updating method with resampling processing](https://readpaper.com/pdf-annotate/note?pdfId=2121926335166128896&noteId=2121926748338632704) Yanlin Zhao, Zhongmin Deng ⇑, Xinjie Zhang

为了更好地估计参数的不确定性，提出了一种鲁棒随机模型更新框架。在该框架中，为了提高鲁棒性，重采样过程主要设计用于处理不良样本点，特别是**有限样本量问题**。其次，**提出了基于巴塔查里亚距离和欧几里得距离的平均距离不确定度UQ度量**，以充分利用测量的可用信息。随后**采用粒子群优化算法**对结构的输入参数进行更新。最后以**质量-弹簧系统和钢板结构为例**说明了该方法的有效性和优越性。通过将测量样品加入不良样品，讨论了重采样过程的作用。

- 有限元模型具有一定的不确定性，主要是由于模型的简化、近似以及模型参数的不确定性，如弹性模量、几何尺寸、边界条件、静、动载荷条件等。
- 实验系统的测量都具有一定的不确定度。这种不确定性与难以控制的随机实验效应有关，如制造公差引入的偏差，随后信号处理期间的测量噪声或有限的测量数据。

算例：
- a mass-spring system
- steel plate structures

## A frequency response model updating method based on unidirectional convolutional neural network

[A frequency response model updating method based on unidirectional convolutional neural network (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=2110304823896008192&noteId=2110305055136380672) Xinjie Zhang , Zhongmin Deng, and Yanlin Zhao 

本文提出了一种基于**单向卷积神经网络** (UCNN) 的方法，以利用频率响应 (FR) 数据进行有限元模型更新。UCNN 旨在在没有任何人工特征提取的情况下从 FR 数据获取高精度逆映射到更新参数。单向卷积分别应用于 FR 数据的频率和位置维度，以避免数据耦合。UCNN 在**卫星模型更新**实验中优于基于残差的模型更新方法和二维卷积神经网络。它在训练集中内外都实现了高精度的结果。

评价指标：
- FR 保证准则(assurance criterion) $FRAC=\frac{|H_e^TH_s|^2}{(H_e^TH_e)(H_s^TH_s)}$ 用于描述试验数据与模拟 FR 数据的相似度

算例：
- 卫星算例1

## Frequency response function-based model updating using Kriging model

提出了一种基于加速度频响函数(FRF)的模型更新方法，该方法**将Kriging模型作为元模型引入到优化过程**中，而不是直接迭代有限元分析。Kriging模型是一种快速运行的模型，可以减少求解时间，便于智能算法在模型更新中的应用。Kriging模型的训练样本由实验设计(DOE)生成，其响应对应于选定频率点上实验加速度频响与有限元模型对应频响的差值。考虑边界条件，**提出了一种减少训练样本数量的两步DOE方法**。第一步是从边界条件中选择设计变量，选择的变量将传递到第二步生成训练样本。将设计变量的优化结果作为设计变量的更新值对有限元进行标定，使解析频响与实验频响趋于一致。该方法在**蜂窝夹层梁复合材料结构**上进行了成功的实验，在模型更新后，分析加速度频域有了明显的改善，特别是在调整阻尼比后，分析加速度频域与实验数据吻合较好。

算例：
- experimental honeycomb beam

# 区间模型修正

## Interval model updating using perturbation method and Radial Basis Function neural networks

[Interval model updating using perturbation method and Radial Basis Function neural networks](https://readpaper.com/pdf-annotate/note?pdfId=2202475379254937088&noteId=2202475618179336192) Zhongmin Deng, Zhaopu Guo n, Xinjie Zhang 北航

近年来，随机模型更新技术已被应用于实际工程结构中固有不确定性的量化。然而在工程实践中，由于结构系统的信息不足，往往无法得到结构参数的概率密度函数。在这种情况下，区间分析在处理不确定问题时显示出明显的优势，因为只定义了输入和输出的上界和下界。为此，**提出了一种利用一阶摄动法和径向基函数神经网络进行结构参数区间辨识的新方法**。通过摄动方法，将每个随机变量表示为每个参数区间均值周围的摄动，这些项可以在两步确定性更新意义上使用。然后在摄动技术的基础上建立了区间模型更新方程。采用两步法对结构参数的均值进行更新，进而对区间半径进行估计。实验和数值算例验证了该方法在结构参数区间识别中的应用。

算例：
- Experimental case study: steel plate structures 钢板
- Numerical case study: Satellite structure

## Interval identification of structural parameters using interval overlap ratio and Monte Carlo simulation

[Interval identification of structural parameters using interval overlap ratio and Monte Carlo simulation](https://readpaper.com/pdf-annotate/note?pdfId=2201610607974204928&noteId=2201611039416835840)Deng Zhongmin a , Guo Zhaopu a,b 北航

从不确定性传播和不确定性量化的角度，**提出了一种新的区间有限元模型更新策略**，用于结构参数的区间识别。将蒙特卡罗仿真与surrogate模型相结合，可以有效地获得系统响应的精确区间估计。利用区间长度的概念，构造了一种新的定量指标——**区间重叠率(IOR)**，用来表征分析数据与实测数据区间分布的一致性。构造并解决了两个不确定结构参数标称值和区间半径估计的优化问题。最后，**通过数值和实验**验证了该方法在结构参数区间识别中的可行性。

不确定性分析问题一般可分为两个方面:不确定性传播和不确定性量化
- 不确定性传播的主要区间方法有区间算法[24]、摄动法[25]、顶点法[8]和MC模拟
  - 各种方法的缺陷：由于忽略了操作数之间的相关性，包含一组操作的区间算法往往高估了系统响应的范围。顶点方法仅对输入和输出之间存在单调关系的特殊情况有效。摄动法虽然考虑了计算效率，但它依赖于区间变量的初值和不确定水平。MC仿真由于精度高、过程简单，被认为是求解输入/输出变量不确定性传播的一种合适的方法。然而，庞大的计算量阻碍了直接MC模拟的实现。
- 不确定度量化的目的是提供一种定量度量，以表征分析预测和测量观测之间的区间范围的一致性。到目前为止，常用的区间模型更新定量指标是基于区间界[8,22]、区间半径[23]或区间交集[26,27]来构建的

算例：
- Numerical case studies: a mass-spring system具有良好分离和紧密模态的三自由度质量-弹簧系统
  - well-separated modes
  - close modes
- Experimental case study: steel plate structures

## Interval Identification of Structural Parameters Using Interval Deviation Degree and Monte Carlo Simulation

[Interval Identification of Structural Parameters Using Interval Deviation Degree and Monte Carlo Simulation](https://readpaper.com/pdf-annotate/note?pdfId=2202709185245307648&noteId=2202709381471330048) Zhaopu Guo and Zhongmin Deng∗ 北航

本文从不确定性传播和不确定性量化两个方面研究了结构参数的区间辨识问题。将蒙特卡罗(MC)模拟与代理模型相结合，可以有效地获得结构响应的精确区间估计。利用区间长度的概念，**构造了区间偏差度(IDD)来表征解析模态数据与实测模态数据间区间分布的不一致**。通过求解两个优化问题，很好地估计了系统参数的标称值和区间半径。最后，通过数值和实验验证了该方法在结构参数区间识别中的可行性。

算例：
- Numerical Case Studies: A Mass-Spring System
- Experimental Case Study: Steel Plate Structures

## Bayesian inversion for imprecise probabilistic models using a novel entropy-based uncertainty quantification metric

[Bayesian inversion for imprecise probabilistic models using a novel entropy-based uncertainty quantification metric](https://readpaper.com/pdf-annotate/note?pdfId=2201700148278027008&noteId=2201703193291922688) Lechang Yang a,b,⇑, Sifeng Bi c , Matthias G.R. Faes b,d , Matteo Broggi b , Michael Beer b,e,f
a北京科技大学机械工程学院，北京100083，中国
b莱布尼茨风险与可靠性研究所Universität汉诺威，德国汉诺威30167
c北京理工大学航天工程学院，北京100081
d鲁汶大学机械工程系，Jan De Nayerlaan 5，比利时
e利物浦大学风险与不确定性研究所，英国利物浦L69 7ZF
f同济大学工程可靠性与随机力学国际联合研究中心，上海200092

不确定性量化度量在动态系统反问题中具有关键地位，因为它们量化了数值预测样本与收集到的观测值之间的差异。这种度量通过奖励这种差异规范较小的样本和惩罚其他样本来发挥其作用。在本文中，我们利用Jensen-Shannon散度提出了一个**新的基于熵的度量**。与其他现有的基于距离的度量相比，一些独特的性质使这种基于熵的度量成为解决混合不确定性(即，不确定性和认知不确定性的组合)存在的逆问题的有效工具，例如在不精确概率的背景下遇到的逆问题。在实现方面，**开发了一种近似贝叶斯计算方法**，其中所提议的度量被完全嵌入。为了减少计算量，采用离散化的分簇算法代替传统的多变量核密度估计。为了验证目的，首先演示了一个静态案例研究，其中与其他三种行之有效的方法进行了比较。为了突出其在复杂动态系统中的潜力，我们将我们的方法应用于**2014年NASA LaRC不确定性量化挑战问题**，并将所获得的结果与文献中其他6个研究小组的结果进行了比较。这些例子说明了我们的方法在静态和动态系统中的有效性，并显示了它在实际工程案例中的前景，如结构健康监测与动态分析相结合。

算例：
- Simply supported beam简支梁
- Practical application example: NASA UQ challenge problem2014

## A novel interval model updating framework based on correlation propagation and matrix-similarity method

[A novel interval model updating framework based on correlation propagation and matrix-similarity method](https://readpaper.com/pdf-annotate/note?pdfId=2202716203238606336&noteId=2202716355810672896) Baopeng Liao , Rui Zhao , Kaiping Yu , Chaoran Liu 哈工大

模型更新技术在实际系统中具有不确定性的数值模型中得到了广泛的应用，而随机理论在知识不足的情况下是无效的。此外，面对相关的不确定性和复杂的数值模型，模型更新仍然是一个挑战。本文提出了一种新的区间模型更新框架，以解决极限样本的相关不确定性问题。这种框架的优点是，**无论输入输出关系是线性的还是非线性的**，都可以高精度地更新参数。为了实现这一优势，**采用凸建模技术和Chebyshev代理模型**分别进行不确定参数量化和数值模型逼近。随后，**提出了考虑关联传播的矩阵相似法**，构建了两步区间模型更新过程，将其转化为确定性模型更新问题。精神的结果。值得注意的是，三个例子验证了所提出的框架在线性和非线性关系中的有效性和优越性。结果表明，本文提出的区间模型更新框架适用于处理参数边界及其相关性的更新问题。

算例：两个数值算例和一个实验算例验证
- a classical mass-spring system
- the composite beam structure
- the real physical system through the experiments of beam structure with sliding masses 带滑动质量的梁结构


## Interval model updating using universal grey mathematics and Gaussian process regression model

[Interval model updating using universal grey mathematics and Gaussian process regression model](https://readpaper.com/pdf-annotate/note?pdfId=2202727212984616448&noteId=2202727326381885440) Bowen Zheng, Kaiping Yu ⇑, Shuaishuai Liu, Rui Zhao 哈工大

本文提出了一种利用通用灰色数学和高斯过程回归模型(GPRM)处理区间模型更新问题的新方法。在区间模型更新问题中，通常存在输入与输出之间的非线性关系，目前的文献没有对此进行专门的讨论。为了更好地处理这一问题，**提出了关注非线性单调关系的方法**。利用通用灰色数学在确定性框架下处理区间模型的更新问题。采用GPRMs作为元模型，提高了计算效率。文中给出了两个数值算例和一个实验算例:第一个数值算例采用三自由度弹簧-质量系统对该方法进行了验证;在第二个数值算例中，以一组铝合金板为例验证了该方法处理非线性问题的能力;最后以一组铝合金板的实验为例说明了该方法，其中以铝合金板的厚度作为更新参数。

算例：
- Numeric case studies 1: mass-spring system
- Numerical case study 2: aluminum alloy plate 
- Experimental validation两块自由铝合金板 固有频率