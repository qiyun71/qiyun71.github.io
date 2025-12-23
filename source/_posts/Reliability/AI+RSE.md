---
title: AI+RSE
date: 2025-10-13 21:16:23
tags: 
categories: Reliability
Year: 
Journal:
---


<!-- more -->

为了提高工程分析和设计的效率，有限元（Finite Element， FE）数值仿真分析在各类工程领域中变得越来越普遍。FE的输入包括结构参数(如几何形状、材料属性)和环境参数(如载荷、边界条件)，输出通常是结构的响应(如位移、应力、频率等)。然而对于复杂的工程系统，FE模型通常计算量大，单次仿真可能需要数小时甚至数天才能完成。这在需要大量重复计算的任务中，如优化设计、可靠性分析和不确定性量化，成为一个主要瓶颈。

随着人工智能技术(Artificial Intelligence, AI）的发展，机器学习(ML)和深度学习(DL)方法被广泛应用于构建FE模型的代理模型。这些代理模型通过学习FE模型的输入输出关系，可以在极短的时间内预测结构响应，从而显著减少计算时间。常见的代理模型包括高斯过程回归(GPR)、支持向量机(SVM)、神经网络(NN)等。

在模型修正领域，主要可以分为基于频率的和基于贝叶斯的方法。
基于频率的方法通过比较FE模型和实验测量的频率响应，调整模型参数以最小化两者之间的差异。常用的方法包括遗传算法(GA)、粒子群优化(PSO)等优化算法。AI在频率学派中的应用除了直接构建从参数到响应的代理模型外，一些学者还提出了构建从响应到参数的反向代理模型。这种反向模型可以直接从测量数据中预测结构参数，避免了复杂的优化过程，提高了计算效率。
基于贝叶斯的方法则利用贝叶斯推断框架，通过结合先验信息和观测数据，更新模型参数的概率分布。这种方法不仅提供了参数的最优估计，还能量化参数的不确定性。AI技术在贝叶斯学派中除了用于构建从参数到响应的代理模型，从而加速近似贝叶斯(Approximate Bayesian Computation, ABC)中直接似然函数的计算外，还可以用于观测响应数据到潜在空间的映射，从而实现似然函数的间接计算。
另外最近随着生成式AI(Generative AI)的发展，一些学者开始探索利用生成式模型(如生成对抗网络GAN、变分自编码器VAE等)来构建更复杂的代理模型。例如Flow-based models和Diffusion models等生成式模型以响应数据为条件，实现了从潜在空间到参数空间的双向映射。

此外在负载辨识和损伤识别等领域，AI技术的应用也体现了反向代理模型的思想。通过学习从结构响应到外部载荷或损伤状态的映射关系，可以直接从测量数据中识别出作用在结构上的载荷或损伤位置和程度，避免了传统方法中复杂的优化过程。


## 研究方向

可靠性系统工程(Reliability System Engineering, RSE)：“运用系统工程理论方法，以故障为核心，以效能为目标，研究复杂系统全寿命过程中故障发生规律及其预防、诊断、修复的综合交叉技术和管理”
- 三全质量观：将产品质量特性划分为功能性能对应的专用质量特性（Special Quality Characteristics，SQC）以及可靠性、安全性、维修性、测试性、保障性、环境适应性（简称六性）等特性对应的通用质量特性（General Quality Characteristics，GQC）
    - 全系统：不同尺度的软/硬件物理实体，如体系、装备、系统、设备、元器件、原材料等
    - 全寿命：论证、设计、试验、生产、评估、验证和使用
    - 全特性：可靠性、安全性、测试性、维修性、保障性、环境适应性等通用质量特性方法

*故障相较于损伤来说，更加广义，既可以指结构中的物理损伤(如裂纹、腐蚀等)，也可以指系统功能的异常(如传感器故障、控制系统失效等)。而损伤通常特指结构中的物理损伤。*

> 故障认知理论主要包括：这些故障认知理论的物理/数学基础主要包括确定性和不确定理论及其综合理论。
>     1) 基于载荷响应和理化过程的故障物理，主要描述结构、材料、工艺对载荷的响应和物理化学变化，如断裂、击穿、疲劳等；
>     2) 基于静态、动态逻辑和涌现行为的失败事理，主要描述系统在任务和决策中的时序以及静态、动态逻辑和涌现关系，如逻辑漏洞和时序错乱等；
>     3) 基于绩效影响和能力局限的失误人理，主要描述机器、环境、团队对人的绩效影响和人的能力局限性，如超负荷、决策错误、误操作等。
> 
> 故障预防技术是指面向产品设计、制造和使用的设计、分析、试验与评价技术。常见的冗余设计、简化设计、基于统计的过程控制技术和以可靠性为中心的维修维护等都属于故障预防技术。
> 故障诊断技术是指对产品故障进行诊断和预测的技术，它侧重于及时地开展故障监测与隔离，关注故障发展趋势和后果的预测。
> 在故障诊断技术的基础上，故障“治疗”技术是指一旦发生不可控故障，及时有效地恢复产品功能的技术。它旨在快速、经济、有效地恢复产品功能，包括修复产品故障的具体技术、修复产品故障的程序以及各类保障资源的规划，如备件、工具、设备，以及修复产品故障所需的人员。
> 
> 应用技术层：基于效能仿真的**需求**集成技术，基于模型驱动的**研制**集成技术以及基于故障预测与健康管理（Prognostics and Health Management，PHM）的**运维**集成技术

![orzsn2pq-f3qw-6kzs-t7gj-d8mtsxij (832×403)|666](https://cdn.linkresearcher.com/orzsn2pq-f3qw-6kzs-t7gj-d8mtsxij)

可靠性系统工程在航空航天、机械、土木等许多领域，都扮演着至关重要的角色。(全系统)大到装备系统，小到结构零部件，其可靠性直接关系到系统的安全性和性能。(全寿命)从生命周期来看，可靠性系统工程涵盖了设计、制造、使用和维护等各个阶段，在设计时预防故障的发生，在使用中监测和评估系统状态，并在维护中采取有效措施延长系统寿命。
- 设计-预防：可靠性设计与优化(Reliability-based design optimization, RBDO)、优化设计(Optimization design)、故障预测与健康管理(Prognostics and Health Management, PHM)、可靠性分析(Reliability analysis)、不确定性量化(Uncertainty quantification)
- 使用-监测：结构健康监测（Structural Health Monitoring, SHM）、损伤检测(Damage detection)、损伤识别(Damage identification)、损伤定位(Damage localization)、负载辨识(Load identification)、模型修正(Model upating/calibration/identification)、故障诊断(Fault diagnosis)、故障检测(Fault detection)、损伤分类(Damage classification)
- 维护-管理：故障预测与健康管理(Prognostics and Health Management, PHM)、损伤量化(Damage quantification)、剩余寿命预测(Remaining Useful Life, RUL)$\approx$健康诊断(Health diagnosis)、基于状态的维护(Condition-based maintenance, CBM)、预测性维护(Predictive maintenance, PdM)、风险评估与管理(Risk assessment and management)、人工自愈（Artificial Self-recovery）

SHM与PHM的异同点：
- 故障预测与健康管理（Prognostics and Health Management, PHM）则侧重于预测结构或系统的未来状态，评估其剩余寿命，并制定维护策略。PHM通常结合物理模型和数据驱动方法，通过分析历史数据和实时监测数据，预测潜在的故障和失效模式。
- 结构健康监测（Structural Health Monitoring, SHM）是利用传感器技术和数据分析方法，对结构的状态进行实时监测和评估。SHM系统通常包括传感器布置、数据采集、信号处理和特征提取等环节。通过分析传感器数据，可以识别结构中的损伤和异常，评估其健康状态。
- 相同点：两者都旨在监测和评估结构或系统的状态，利用传感器数据进行分析。
- 不同点：SHM侧重于当前状态的监测和评估，而PHM侧重于未来状态的预测和寿命评估，以及维护策略的制定。

研究方向：
- 模型修正(Model upating/calibration/identification)：寻找最优的结构参数，使得有限元的仿真响应与真实测量实验响应尽可能接近。
- 损伤识别(Damage identification)：根据测量数据，识别结构中的损伤位置和程度。
- 损伤检测(Damage detection)：检测结构中是否存在损伤或异常。
- 负载辨识(Load identification)：根据测量数据，识别作用在结构上的外部载荷。
- 优化设计(Optimization design)：在满足约束条件下，寻找最佳的结构参数组合，以优化结构性能。
- 可靠性分析(Reliability analysis)：评估结构在不确定性条件下的性能和失效概率。
- 不确定性量化(Uncertainty quantification)：评估输入参数的不确定性对结构响应的影响。
- 可靠性设计与优化(Reliability-based design optimization, RBDO)：结合可靠性分析和优化设计，寻找在满足可靠性要求下的最佳结构参数组合（如成本、重量等）。

不太熟悉：
- 结构动力学分析(Structural dynamics analysis)：研究结构在动态载荷作用下的响应和行为。
- 多尺度建模(Multiscale modeling)：结合不同尺度的模型，研究材料和结构的行为。
- 多物理场耦合(Multiphysics coupling)：研究不同物理场（如热、力、电等）之间的相互作用对结构性能的影响。
- 实时监测与控制(Real-time monitoring and control)：开发实时监测系统，结合控制策略，提高结构的安全性和性能。
- 数据同化(Data assimilation)：结合观测数据和数值模型，提高对结构状态的估计精度。
- 传感器布置优化(Sensor placement optimization)：设计最优的传感器布局，以提高监测系统的效率和准确性。
- 大数据分析(Big data analytics)：利用大数据技术，处理和分析大量传感器数据，提取有用信息。
- 边界条件识别(Boundary condition identification)：根据测量数据，识别结构的边界条件。




## Follow

[Zili WANG | Beihang University, Beijing | BUAA | Research profile](https://www.researchgate.net/profile/Zili-Wang-3)

## AI-RSE

AI > Machine Learning > Deep Learning

- model-based:
- data-driven: 

Machine Learning:
- Supervised learning
    - *discrete* classification 
    - *continuous* regression
- Unsupervised learning
    - *discrete* clustering 
    - *continuous* dimensionality reduction
    - Anomaly detection
- Semi-supervised learning
    - *discrete* classification 
    - *continuous* regression
- Reinforcement learning
    - model-free
    - model-based


What's the machine learning classified as?

“PHM-LM” ([Tao 等, 2025, p. 15](zotero://select/library/items/EA4UD3UT)) ([pdf](zotero://open-pdf/library/items/GFXBIP9I?page=15&annotation=767ZF8XK))

“ML can be classified into three major categories, with a fourth one straddling the first two: (i) supervised learning; (ii) unsupervised learning; (iii) semisupervised learning; and (iv) reinforcement learning.” ([Xu和Saleh, 2021, p. 2](zotero://select/library/items/UY4D3ZYI)) ([pdf](zotero://open-pdf/library/items/EQTBIFYL?page=2&annotation=MXMYLHMZ))

Applications:

RUL: remaining useful life estimation/prediction/prognostic
- model-based approach: build failure models based on the detailed analysis of the physical nature of the failure mechanism under consideration
- data-driven approach: which build degradation models from historical sensor data, and therefore require no prior knowledge of the system

structural reliability analysis is to estimate failure probability 


### Supervised learning
#### Regression

| Article                                                                                                                                                                                                                                                    | Method                                                                                             | Application                                                                             |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| [Chang et al., A hybrid prognostic method for system degradation based on particle filter and relevance vector machine, 2019-06-01, Reliability Engineering & System Safety](zotero://select/library/items/JYGPQTFY)                                       | combines particle filter (PF) and relevance vector machine (RVM)                                   | RUL                                                                                     |
| [Liu et al., Remaining useful life prediction based on health index similarity, 2019-05-01, Reliability Engineering & System Safety](zotero://select/library/items/7VV6WU4H)                                                                               | health index (HI)                                                                                  | RUL                                                                                     |
| [Fink et al., Predicting component reliability and level of degradation with complex-valued neural networks, 2014-01-01, Reliability Engineering & System Safety](zotero://select/library/items/Z6Y3H3PH)                                                  | DNN                                                                                                | reliability and level of degradation prediction                                         |
| [Li et al., Deep learning-based remaining useful life estimation of bearings using multi-scale feature extraction, 2019-02-01, Reliability Engineering & System Safety](zotero://select/library/items/4EWFFXCX)                                            | CNN                                                                                                | RUL                                                                                     |
| [García Nieto et al., Hybrid PSO–SVM-based method for forecasting of the remaining useful life for aircraft engines and evaluation of its reliability, 2015-06-01, Reliability Engineering & System Safety](zotero://select/library/items/4YNB7XD9)        | PSO–SVM                                                                                            | RUL                                                                                     |
| [Zhao et al., Remaining useful life prediction of aircraft engine based on degradation pattern learning, 2017-08-01, Reliability Engineering & System Safety](zotero://select/library/items/8M4C7LUH)                                                      | DNN                                                                                                | RUL                                                                                     |
| [Ling et al., Efficient methods by active learning Kriging coupled with variance reduction based sampling methods for time-dependent failure probability, 2019-08-01, Reliability Engineering & System Safety](zotero://select/library/items/LS8SDMUI)     | AK coupled with importance sampling (AK-co-IS)<br>AK coupled with subset simulation (AK-co-SS)<br> | time-dependent failure probability(TDFP)                                                |
| [Moura et al., Failure and reliability prediction by support vector machines regression of time series data, 2011-11-01, Reliability Engineering & System Safety](zotero://select/library/items/T2MR4IFI)                                                  | Support Vector Machines (SVMs)                                                                     | Failure and reliability prediction                                                      |
| [Worrell et al., Machine learning of fire hazard model simulations for use in probabilistic safety assessments at nuclear power plants, 2019-03-01, Reliability Engineering & System Safety](zotero://select/library/items/DCV79FM5)                       | Metamodels<br> k-nearest neighbor (kNN)                                                            | probabilistic safety assessments<br>approximations of a physics-based fire hazard model |
| [Vanderhaegen et al., A Benefit/Cost/Deficit (BCD) model for learning from human errors, 2011-07-01, Reliability Engineering & System Safety](zotero://select/library/items/5VULIBTT)                                                                      | A Benefit/Cost/Deficit (BCD) model<br>a case-based reasoning system and a neural network system    | interpreting human errors                                                               |
| [Li et al., Remaining useful life estimation in prognostics using deep convolution neural networks, 2018-04-01, Reliability Engineering & System Safety](zotero://select/library/items/I9BSRJNW)                                                           | CNN                                                                                                | RUL                                                                                     |
| [Heimes, Recurrent neural networks for remaining useful life estimation, 2008-10, 2008 International Conference on Prognostics and Health Management](zotero://select/library/items/43FABPQH)                                                              | RNN                                                                                                | RUL                                                                                     |
| [Wang et al., Remaining Useful Life Estimation in Prognostics Using Deep Bidirectional LSTM Neural Network, 2018-10, 2018 Prognostics and System Health Management Conference (phm-chongqing)](zotero://select/library/items/PAF48MEU)                     | LSTM                                                                                               | RUL                                                                                     |
| [Chen et al., Gated recurrent unit based recurrent neural network for remaining useful life prediction of nonlinear deterioration process, 2019-05-01, Reliability Engineering & System Safety](zotero://select/library/items/IU76QB9U)                    | kernel principle component analysis (KPCA)<br>LSTM+gated recurrent unit (GRU)                      | RUL                                                                                     |
| [Sun et al., LIF: A new Kriging based learning function and its application to structural reliability analysis, 2017-01-01, Reliability Engineering & System Safety](zotero://select/library/items/75U55VUH)                                               | Kriging                                                                                            | structural reliability analysis                                                         |
| [Zhang et al., An active learning reliability method combining Kriging constructed with exploration and exploitation of failure region and subset simulation, 2019-08-01, Reliability Engineering & System Safety](zotero://select/library/items/3PT8IXQ6) |  active learning Kriging (AK)+Subset simulation (SS)                                               | structural reliability analysis                                                         |
| [Yang et al., System reliability analysis through active learning Kriging model with truncated candidate region, 2018-01-01, Reliability Engineering & System Safety](zotero://select/library/items/Q7BSTFM2)                                              | AK<br>truncated candidate region (TCR)                                                             | System reliability analysis                                                             |
| [Jiang et al., A general failure-pursuing sampling framework for surrogate-based reliability analysis, 2019-03-01, Reliability Engineering & System Safety](zotero://select/library/items/EIM3U7B4)                                                        | AK                                                                                                 | reliability analysis                                                                    |
| [Wei et al., Reliability and reliability-based importance analysis of structural systems using multiple response Gaussian process model, 2018-07-01, Reliability Engineering & System Safety](zotero://select/library/items/CG2YVDHP)                      | AK + MCS<br>variable importance analysis (VIA) indices<br>mode importance analysis (MIA) indices   | Reliability-based importance analysis of structural system                              |
| [Pecht et al., Physics-of-failure-based prognostics for electronic products, 2009-06, Transactions of the Institute of Measurement and Control](zotero://select/library/items/3YLQAT2S)                                                                    | physics-of-failure (PoF)-based PHM method<br>**model-based**                                       | reliability prediction<br>reliability design and<br>assessment. P                       |
|                                                                                                                                                                                                                                                            |                                                                                                    |                                                                                         |


#### Classification


| Article                                                                                                                                                                                                                                                | Method                                                                                                                                                                                            | Application                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| [Tamilselvan et al., Failure diagnosis using deep belief learning based health state classification, 2013-07-01, Reliability Engineering & System Safety](zotero://select/library/items/LHHX47ZI)                                                      | deep belief network (DBN)                                                                                                                                                                         | Failure diagnosis                                           |
| [Tang et al., Comparison of automatic and guided learning for Bayesian networks to analyse pipe failures in the water distribution system, 2019-06-01, Reliability Engineering & System Safety](zotero://select/library/items/74PJ972Q)                | Bayesian networks                                                                                                                                                                                 | analyse pipe failures in the water distribution system      |
| [Manjurul Islam et al., Reliable multiple combined fault diagnosis of bearings using heterogeneous feature models and multiclass support vector Machines, 2019-04-01, Reliability Engineering & System Safety](zotero://select/library/items/VWYZCYTS) | one-against-all multiclass **support vector machines** (OAA-MCSVM)                                                                                                                                | fault diagnosis                                             |
| [Tao et al., Bearing fault diagnosis method based on stacked autoencoder and softmax regression, 2015-07, 2015 34th Chinese Control Conference (CCC)](zotero://select/library/items/ZC7SHBMJ)                                                          | DNN                                                                                                                                                                                               | fault diagnosis                                             |
| [Arcos Jiménez et al., Dirt and mud detection and diagnosis on a wind turbine blade employing guided waves and supervised learning classifiers, 2019-04, Reliability Engineering & System Safety](zotero://select/library/items/PQDFP4LC)              | supervised learning classifiers<br>Ensemble Subspace Discriminant; k-Nearest Neighbours; Linear Support Vector Machine; Linear Discriminant Analysis; Decision Trees                              | Dirt and Mud Detection and Diagnosis                        |
| [Wang et al., Software reliability prediction using a deep learning model based on the RNN encoder–decoder, 2018-02-01, Reliability Engineering & System Safety](zotero://select/library/items/45756S4R)                                               | RNN                                                                                                                                                                                               | Software reliability prediction                             |
| [Naderpour et al., Forest fire induced Natech risk assessment: A survey of geospatial technologies, 2019-11-01, Reliability Engineering & System Safety](zotero://select/library/items/IQCBDMML)                                                       | (a) statistical and data-driven models; (b) machine learning models; (c) multi-criteria decision-making models, and (d) ensemble models.                                                          | Natech risk assessment                                      |
| [Stern et al., Accelerated Monte Carlo system reliability analysis through machine-learning-based surrogate models of network connectivity, 2017-08-01, Reliability Engineering & System Safety](zotero://select/library/items/R859DAE4)               | a support vector machine (SVM) and a logistic regression                                                                                                                                          | network connectivity                                        |
| [Rachman et al., Machine learning approach for risk-based inspection screening assessment, 2019-05-01, Reliability Engineering & System Safety](zotero://select/library/items/VE4IQCDV)                                                                | Machine learning approach<br>logistic regression (LR), support vector machines (SVM), k-nearest neighbors (k-NN), gradient boosting decision trees (GBDT), AdaBoost (AB), and random forests (RF) | risk-based inspection screening assessment                  |
| [Hernandez-Perdomo et al., A reliability model for assessing corporate governance using machine learning techniques, 2019-05-01, Reliability Engineering & System Safety](zotero://select/library/items/B4D4XMQB)                                      | Decision tree                                                                                                                                                                                     | Assessing stakeholders corporate governance                 |
| [Gehl et al., Approximate Bayesian network formulation for the rapid loss assessment of real-world infrastructure systems, 2018-09-01, Reliability Engineering & System Safety](zotero://select/library/items/NWNUXGUE)                                | approximate Bayesian network (BN)                                                                                                                                                                 | Rank the importance of components of the engineering system |
| [Marseguerra, Early detection of gradual concept drifts by text categorization and Support Vector Machine techniques: The TRIO algorithm, 2014-09-01, Reliability Engineering & System Safety](zotero://select/library/items/BHGZZP5P)                 | Support Vector Machine                                                                                                                                                                            | Early detection of gradual concept drifts                   |
| [Gui et al., Data-driven support vector machine with optimization techniques for structural health monitoring and damage detection, 2017-02-01, KSCE Journal of Civil Engineering](zotero://select/library/items/MLQ4Z28P)                             | three optimization-algorithm based support vector machines<br>grid-search, partial swarm optimization and genetic algorithm                                                                       | structural health monitoring and damage detection           |
| [Bao et al., Structural damage detection based on non-negative matrix factorization and relevance vector machine, 2010-03-25](zotero://select/library/items/VCADXLFD)                                                                                  | relevance vector machine (RVM)                                                                                                                                                                    | Structural damage detection                                 |
| [Perrin, Active learning surrogate models for the conception of systems with multiple failure modes, 2016-05-01, Reliability Engineering & System Safety](zotero://select/library/items/L6UCNTPQ)                                                      | nested Gaussian process surrogate models                                                                                                                                                          | Reliability evaluation of complex system                    |
| [Xiang et al., An active learning method combining deep neural network and weighted sampling for structural reliability analysis, 2020-06-01, Mechanical Systems and Signal Processing](zotero://select/library/items/ZBIBSMSJ)                        | DNN                                                                                                                                                                                               | structural reliability analysis                             |
| [Nguyen et al., A new dynamic predictive maintenance framework using deep learning for failure prognostics, 2019-08-01, Reliability Engineering & System Safety](zotero://select/library/items/E3KXR83X)                                               | LSTM                                                                                                                                                                                              | failure prognostics                                         |


### Unsupervised learning

#### Clustering


| Article                                                                                                                                                                                                                                                                          | Method                                                                | Application                                                             |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [Bian et al., Degradation state mining and identification for railway point machines, 2019-08-01, Reliability Engineering & System Safety](zotero://select/library/items/QTTBQ4RA)                                                                                               | self-organizing feature mapping and support vector machine            | Degradation state identificationEarly fault diagnosis                   |
| [Reder et al., Data-driven learning framework for associating weather conditions and wind turbine failures, 2018-01-01, Reliability Engineering & System Safety](zotero://select/library/items/8RBIGASK)                                                                         | k-means                                                               | associating weather conditions and wind turbine failures                |
| [Sirola et al., SOM based methods in early fault detection of nuclear industry, 2009](zotero://select/library/items/CTF43YLW)                                                                                                                                                    | Self-Organizing Map (SOM)                                             | Early fault detection                                                   |
| [Tibaduiza et al., Damage classification in structural health monitoring using principal component analysis and self-organizing maps: DAMAGE CLASSIFICATION IN SHM USING PCA AND SOM, 2013-10, Structural Control and Health Monitoring](zotero://select/library/items/YTKBIGBM) | principal<br>component analysis (PCA) and self-organizing maps (SOM)  | Damage classification in structural health monitoring                   |
| [Zhou et al., Bearing fault recognition method based on neighbourhood component analysis and coupled hidden Markov model, 2016-01-01, Mechanical Systems and Signal Processing](zotero://select/library/items/UVH3M7IX)                                                          | Neighbourhood component analysis (NCA)<br>coupled hidden Markov model | dimensionality reduction and feature extraction<br><br> fault diagnosis |
| [Gerassis et al., Understanding complex blasting operations: A structural equation model combining Bayesian networks and latent class clustering, 2019-08-01, Reliability Engineering & System Safety](zotero://select/library/items/A42GE2HG)                                   | Bayesian networks and latent class clustering                         | Risk analysis                                                           |
| [Fang et al., Unsupervised spectral clustering for hierarchical modelling and criticality analysis of complex networks, 2013-08-01, Reliability Engineering & System Safety](zotero://select/library/items/4C4UCNJG)                                                             | Spectral clustering                                                   | criticality analysis of complex networks                                |
| [Soualhi et al., Detection and Diagnosis of Faults in Induction Motor Using an Improved Artificial Ant Clustering Technique, 2013-09, IEEE Transactions on Industrial Electronics](zotero://select/library/items/6MXPCXHY)                                                       | Artificial Ant Clustering                                             | Detection and Diagnosis of Faults                                       |
| [Prabakaran et al., Self-Organizing Map Based Fault Detection and Isolation Scheme for Pneumatic Actuator](zotero://select/library/items/SZ853R5H)                                                                                                                               | Self-Organizing Map                                                   | Fault Detection                                                         |
| [Malhotra et al., Multi-Sensor Prognostics using an Unsupervised Health Index based on LSTM Encoder-Decoder, 2016-08-22](zotero://select/library/items/ZUP9BGEL)                                                                                                                 | LSTM                                                                  | Multi-Sensor Prognostics                                                |

#### Dimensionality reduction

| Article | Method | Application |
| ------- | ------ | ----------- |
|         |        |             |
|         |        |             |

#### Anomaly Detection

| Article                                                                                                                                                                                                                                                                                                                                                                                                                                    | Method                                       | Application                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------- | ------------------------------- |
| [Reddy et al., Anomaly Detection and Fault Disambiguation in Large Flight Data: A Multi-modal Deep Auto-encoder Approach, 2016-10-03, Annual Conference of the PHM Society](zotero://select/library/items/4H743WV2)                                                                                                                                                                                                                        | Deep Auto-encoders (DAE)                     | fault diagnosis                 |
| [Yan et al., On Accurate and Reliable Anomaly Detection for Gas Turbine Combustors: A Deep Learning Approach, 2019-08-25](zotero://select/library/items/R3DJYHNE)                                                                                                                                                                                                                                                                          | Deep neural network                          | Anomaly Detection               |
| [Fuertes et al., Improving Spacecraft Health Monitoring with Automatic Anomaly Detection Techniques, 2016-05-16, Spaceops 2016 Conference](zotero://select/library/items/RIME59YZ)                                                                                                                                                                                                                                                         | One-Class Support Vector Machine<br>(OC-SVM) | Anomaly Detection               |
| [Hundman et al., Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding, 2018-07-19, Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining](zotero://select/library/items/CIDPGNV4)                                                                                                                                                                                  | LSTM                                         | Detecting Spacecraft Anomalies  |
| [Ince et al., Real-Time Motor Fault Detection by 1-D Convolutional Neural Networks, 2016-11, IEEE Transactions on Industrial Electronics](zotero://select/library/items/UWKMKXSJ)                                                                                                                                                                                                                                                          | 1-D CNN                                      | Real-Time Motor Fault Detection |
| [Schlechtingen et al., Wind turbine condition monitoring based on SCADA data using normal behavior models. Part 1: System description, 2013-01-01, Applied Soft Computing](zotero://select/library/items/9NWJN23L)<br>[Schlechtingen et al., Wind turbine condition monitoring based on SCADA data using normal behavior models. Part 2: Application examples, 2014-01-01, Applied Soft Computing](zotero://select/library/items/BFHFXEQG) | DNN                                          | Condition monitoring            |
| [Souza et al., Evaluation of Data Based Normal Behavior Models for Fault Detection in Wind Turbines, 2019-10, 2019 8th Brazilian Conference on Intelligent Systems (BRACIS)](zotero://select/library/items/XV6H9L2T)                                                                                                                                                                                                                       | DNN                                          | Fault Detection                 |
| [An artificial neural network‐based condition monitoring method for wind turbines, with application to the monitoring of the gearbox](zotero://select/library/items/4ASRA4RK)                                                                                                                                                                                                                                                              | DNN                                          | Condition monitoring            |
| [Garcia et al., SIMAP: Intelligent System for Predictive Maintenance: Application to the health condition monitoring of a windturbine gearbox, 2006-08-01, Computers in Industry](zotero://select/library/items/LFZF84YA)                                                                                                                                                                                                                  |                                              | health condition monitoring     |


### Semi-supervised learning (SSL)

| Article                                                                                                                                                                                                                                         | Method                                                                                  | Application                        |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------- |
| [Hu et al., Semi-supervised learning with co-training for data-driven prognostics, 2012-06, 2012 IEEE Conference on Prognostics and Health Management](zotero://select/library/items/U8DMCIZN)                                                  | a feed-forward neural network (FFNN) approach and a radial basis network (RBN) approach | data-driven prognostics<br>RUL     |
| [He et al., Developing ladder network for intelligent evaluation system: Case of remaining useful life prediction for centrifugal pumps, 2018-12-01, Reliability Engineering & System Safety](zotero://select/library/items/6VNLH2X4)           | **ladder network**                                                                      | RUL                                |
| [Listou Ellefsen et al., Remaining useful life predictions for turbofan engine degradation using semi-supervised deep architecture, 2019-03-01, Reliability Engineering & System Safety](zotero://select/library/items/EMEWSLCR)                | LSTM                                                                                    | RUL                                |
| [Yoon et al., Semi-supervised Learning with Deep Generative Models for Asset Failure Prediction, 2017-09-04](zotero://select/library/items/86QMM2ME)                                                                                            | VAE                                                                                     | Asset Failure Prediction           |
| [Razavi-Far et al., A Semi-Supervised Diagnostic Framework Based on the Surface Estimation of Faulty Distributions, 2019-03, IEEE Transactions on Industrial Informatics](zotero://select/library/items/T7VXJ83R)                               | Semi-supervised Smooth Alpha Layering                                                   | Fault diagnostic                   |
| [Zhao et al., Graph-Based Semi-supervised Learning for Fault Detection and Classification in Solar Photovoltaic Arrays, 2015-05, IEEE Transactions on Power Electronics](zotero://select/library/items/34MWUUGX)                                | Graph based model                                                                       | Fault Detection and Classification |
| [Feng et al., Label consistent semi-supervised non-negative matrix factorization for maintenance activities identification, 2016-06-01, Engineering Applications of Artificial Intelligence](zotero://select/library/items/8RRJ3FCQ)            | Graph based model                                                                       | Health prognostic                  |
| [Aria et al., Near-Miss Accident Detection for Ironworkers Using Inertial Measurement Unit Sensors, 2014-07-08, International Symposium on Automation and Robotics in Construction (ISARC) Proceedings](zotero://select/library/items/77KA8RSV) | One-Class Support Vector Machine (OC-SVM)                                               | Near-Miss Accident Detection       |
|                                                                                                                                                                                                                                                 |                                                                                         |                                    |
|                                                                                                                                                                                                                                                 |                                                                                         |                                    |
### Reinforcement learning (RL)

| Article                                                                                                                                                                                                                                                                              | Method                                                  | Application                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------- | ------------------------------------------------------------------------- |
| [Mohajer et al., Mobility-aware load Balancing for Reliable Self-Organization Networks: Multi-agent Deep Reinforcement Learning, 2020-10-01, Reliability Engineering & System Safety](zotero://select/library/items/APRN7F2W)                                                        | Deep Reinforcement Learning (DRL)                       | mobility robustness optimization (MRO)                                    |
| [Zhang et al., Deep reinforcement learning for condition-based maintenance planning of multi-component systems under dependent competing risks, 2020-11-01, Reliability Engineering & System Safety](zotero://select/library/items/PQ5GYYUP)                                         | Deep reinforcement learning                             | condition-based maintenance planning                                      |
| [Papakonstantinou et al., Optimum inspection and maintenance policies for corroded structures using partially observable Markov decision processes and stochastic, physically based models, 2014-07-01, Probabilistic Engineering Mechanics](zotero://select/library/items/96XPV6JM) | Partially Observable Markov Decision Processes (POMDPs) | risk management and life-cycle cost procedures                            |
| [Andriotis et al., Managing engineering systems with large state and action spaces through deep reinforcement learning, 2019-11-01, Reliability Engineering & System Safety](zotero://select/library/items/EGNVRBIX)                                                                 | <br>Deep reinforcement learning                         | life-cycle engineering systems management                                 |
| [Xiang et al., Deep reinforcement learning-based sampling method for structural reliability assessment, 2020-07-01, Reliability Engineering & System Safety](zotero://select/library/items/YXZRDTUK)                                                                                 | Deep reinforcement learning-based sampling              | structural reliability assessment                                         |
| [Garcıa et al., A Comprehensive Survey on Safe Reinforcement Learning](zotero://select/library/items/BWPCYPT8)                                                                                                                                                                       | Safe Reinforcement Learning Review                      | ensure reasonable<br>system performance and/or respect safety constraints |
|                                                                                                                                                                                                                                                                                      |                                                         |                                                                           |
