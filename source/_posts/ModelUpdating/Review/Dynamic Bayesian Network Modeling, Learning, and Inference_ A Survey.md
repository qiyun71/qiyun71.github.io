---
zotero-key: SVNXIDH6
zt-attachments:
  - "6505"
title: "Dynamic Bayesian Network Modeling, Learning, and Inference: A Survey"
created: 2025-09-10 02:48:16
modified: 2025-09-10 02:48:59
tags: Probability distribution  Bayes methods  Probabilistic logic  Dynamic Bayesian networks  dynamic probabilistic graphical models  Hidden Markov models  literature review  Markov processes  systematic literature review  Systematics
collections: Dynamic Bayesian Network
year: 2021
publication: IEEE Access
citekey: shiguiharaDynamicBayesianNetwork2021
author:
  - Pedro Shiguihara
  - Alneu De Andrade Lopes
  - David Mauricio
---
| Title        | "Dynamic Bayesian Network Modeling, Learning, and Inference: A Survey"                                                                                                                                                                                        |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author       | [Pedro Shiguihara,Alneu De Andrade Lopes,David Mauricio]                                                                                                                                                                                                      |
| Organization |                                                                                                                                                                                                                                                               |
| Paper        | [Zotero pdf](zotero://select/library/items/SVNXIDH6) [attachment](<file:///D:/Download/Zotero_data/storage/J8N6PY27/Shiguihara%20%E7%AD%89%20-%202021%20-%20Dynamic%20Bayesian%20Network%20Modeling,%20Learning,%20and%20Inference%20A%20Survey.pdf>)<br><br> |
| Project      |                                                                                                                                                                                                                                                               |

<!-- more -->

使用监测数据更新(不确定的)疲劳裂纹参数，来判断下次飞行是否可重复

**修正后的裂纹几何依然靠传统的经验因子来实现，而不是直接改模型几何**

## Background

Digital twin, DT-->Airframe DT

主要应用：fatigue crack growth analysis
主要方法：滤波器(**Kalman** filter)、**Dynamic Bayesian network**
- DBN for model updating with time-sequence information
- inference algorithms of DBN include: the Kalman filter [39], the extended Kalman filter [40], the unscented Kalman filter [41], and the particle filters [42] (PF).

“The conventional health management of a spacecraft” ([Ye 等, 2020, p. 2](zotero://select/library/items/XHUM9M9Z)) ([pdf](zotero://open-pdf/library/items/WEFEMB2H?page=2&annotation=RCAQ3E39))
- “(1) providing robust, damage tolerant designs and structural integrity margins to prevent potential failures” ([Ye 等, 2020, p. 2](zotero://select/library/items/XHUM9M9Z)) ([pdf](zotero://open-pdf/library/items/WEFEMB2H?page=2&annotation=UWBUTZQU))
- “(2) relying on scheduled maintenance and lengthy inspections, to identify necessary repairs” ([Ye 等, 2020, p. 2](zotero://select/library/items/XHUM9M9Z)) ([pdf](zotero://open-pdf/library/items/WEFEMB2H?page=2&annotation=EMQ5ZQEF))

但是传统方法存在一些缺陷：
- 复杂的环境和负载、材料特性的不确定性、建模的假设
- 预定义的维修周期没有考虑实际环境和操作，导致飞行过程中早期故障危险。过于频繁的维修又会导致的高成本

DT framework可以解决上述缺陷。分享了ADT解决问题的几个实际案例

- 一些方法证明了DBN的预测和不确定性管理能力，但是仅可用于单次修正
- 融合DBN后可以通过不断增加的观测数据进行在线修正，但将不确定性大的疲劳参数假设为了常数
- 扩展卡尔曼滤波虽然考虑了疲劳参数的不确定性，但是其无法应用到高度非线性系统中

## Innovation

“The key of the framework is to integrate various models, multi-source data, and uncertain parameters into a DBN” ([Ye 等, 2020, p. 2](zotero://select/library/items/XHUM9M9Z)) ([pdf](zotero://open-pdf/library/items/WEFEMB2H?page=2&annotation=PSERP7FA))

proposed digital twin framework:
- “(1) It distinguishes **digital twin functions** (diagnosis, model updating, performance evaluation and data storage) from **digital twin applications** (monitoring, structural health evaluation, prognosis and data sharing).” ([Ye 等, 2020, p. 4](zotero://select/library/items/XHUM9M9Z)) ([pdf](zotero://open-pdf/library/items/WEFEMB2H?page=4&annotation=FN3DIUA3))
- “(2) The framework proposed offline and online stages to separate digital twin development, implementation and maintenance into different groups.” ([Ye 等, 2020, p. 4](zotero://select/library/items/XHUM9M9Z)) ([pdf](zotero://open-pdf/library/items/WEFEMB2H?page=4&annotation=BU4PBFS9))
- “(3) It is centered on the model updating within the uncertainty framework.” ([Ye 等, 2020, p. 4](zotero://select/library/items/XHUM9M9Z)) ([pdf](zotero://open-pdf/library/items/WEFEMB2H?page=4&annotation=2XZQUQ4X))



## Outlook

## Cases

### A demonstration example: a load-bearing airframe/beam

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250910123925.png)


#### Numerical validation example

修正后的数值验证



## Equation

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250910110311.png)

### dynamic Bayesian network （DBN）

inference algorithms of DBN include: 
- the Kalman filter [39], 
- the extended Kalman filter [40], 
- the unscented Kalman filter [41], 
- the particle filters [42] (PF).

“The PF can handle: 1) discrete/continuous variables; 2) Gaussian/nonGaussian distributions; and 3) linear/non-linear relationships, which is well-suited for a general DBN.” ([Ye 等, 2020, p. 6](zotero://select/library/items/XHUM9M9Z)) ([pdf](zotero://open-pdf/library/items/WEFEMB2H?page=6&annotation=MFSLF9D9))
the sequential importance resampling (SIR) algorithm 也是一种PF算法，其他的衍生算法包括：
- regularized particle filter [43], 
- the auxiliary particle filter [44], 
- the Gaussian sum particle filter [45], 
- the reduced order particle filter [46]




kalman filter （KF）




dynamic data driven application system (DDDAS)

“Dynamic Data Driven Application Systems (DDDAS) entails the ability to incorporate additional data into an executing application - these data can be archival or collected on-line; and in reverse, the ability of applications to dynamically steer the measurement process” ([Darema, 2004, p. 662](zotero://select/library/items/EFQ7W4TT)) ([pdf](zotero://open-pdf/library/items/3MIAQ99Y?page=1&annotation=6YPN4P2N))