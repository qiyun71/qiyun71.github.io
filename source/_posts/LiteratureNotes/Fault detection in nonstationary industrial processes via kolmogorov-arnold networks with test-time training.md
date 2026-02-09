---
zotero-key: 2JTFKDQQ
zt-attachments:
  - "7702"
title: Fault detection in nonstationary industrial processes via kolmogorov-arnold networks with test-time training
created: 2025-10-27 08:15:39
modified: 2025-10-27 08:15:40
tags: Fault detection  Industrial process monitoring  Kolmogorov–arnold networks  Nonstationary processes  Online learning
collections: Inverse SM
year: 2025
publication: ISA Transactions
citekey: liFaultDetectionNonstationary2025
author:
  - Daye Li
  - Jie Dong
  - Kaixiang Peng
  - Silvio Simani
  - Chuanfang Zhang
  - Dongjie Hua
---
| Title        | "Fault detection in nonstationary industrial processes via kolmogorov-arnold networks with test-time training"                                                                                                                                                                            |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author       | [Daye Li,Jie Dong,Kaixiang Peng,Silvio Simani,Chuanfang Zhang,Dongjie Hua]                                                                                                                                                                                                                |
| Organization |                                                                                                                                                                                                                                                                                           |
| Paper        | [Zotero pdf](zotero://select/library/items/2JTFKDQQ) [attachment](<file:///D:/Download/Zotero_data/storage/TVC6QHPY/Li%20%E7%AD%89%20-%202025%20-%20Fault%20detection%20in%20nonstationary%20industrial%20processes%20via%20kolmogorov-arnold%20networks%20with%20test-time.pdf>)<br><br> |
| Project      |                                                                                                                                                                                                                                                                                           |

<!-- more -->

## Background

工业过程监测中，production schedule changes, equipment aging, and environmental disturbances等因素会导致shifts in the underlying data distribution。从而增加误警率，危害传统的fault detection方法的可靠性和适应性。

## Innovation

lifelong fault detection策略，“integrates a KolmogorovArnold Network (**KAN**) with a novel test-time training (**TTT**) mechanism” ([Li 等, 2025, p. 1](zotero://select/library/items/2JTFKDQQ)) ([pdf](zotero://open-pdf/library/items/TVC6QHPY?page=4&annotation=SWICXB7B))
- **TKAN**，通过KAN+TTT实现连续model updating、在线调整和动态优化
- **Attention**，“A feature extraction technique that integrates the KAN structure with a multi-head attention mechanism is developed.” ([Li 等, 2025, p. 3](zotero://select/library/items/2JTFKDQQ)) ([pdf](zotero://open-pdf/library/items/TVC6QHPY?page=6&annotation=6ZY28LRF))
- “A **selective strategy** for freezing and updating parameters during the online test phase is introduced.” ([Li 等, 2025, p. 3](zotero://select/library/items/2JTFKDQQ)) ([pdf](zotero://open-pdf/library/items/TVC6QHPY?page=6&annotation=ZHX96XEZ))

KAN相较于MLP，设置了可学习的激活函数，**拥有更强的非线性表达能力和泛化能力？** 

## Outlook

KAN ***训练耗时***


## Cases

## Equation

![1-s2.0-S0019057825005919-gr2_lrg.jpg (1734×1734)|666](https://ars.els-cdn.com/content/image/1-s2.0-S0019057825005919-gr2_lrg.jpg)


1. 在offline train阶段，根据historical data训练AE(Linear encoder + KAN bottleneck and decoder)模型
  - Linear encoder: reduce the time-series data dimension
  - KAN bottleneck: capture the nonlinear feature, feature compression and transform into meaningful latent representations
  - KAN decoder:  reconstruct the original input data from the latent representations
2. 进行statistic in the extracted latent features，并选择合适的控制阈值
  - K-Nearest Neighbours (KNN) approach, based on Euclidean distance： 被用于量化deviations indicative of potential faultsmean Euclidean distance to its K nearest neighbours： 用于根据confidence level $\alpha$ 确定控制阈值
1. 在test-time阶段，利用TTT机制在线更新KAN模型参数
  - Initialisation of TTT mechanism
    - 滑动窗口确保“temporal correlations and dynamics within the data are effectively captured and leveraged” ([Li 等, 2025, p. 11](zotero://select/library/items/2JTFKDQQ)) ([pdf](zotero://open-pdf/library/items/TVC6QHPY?page=11&annotation=ANHNKU3F))
  - Feature Extraction and Model Parameter Update
    - 在Test-time阶段，冻结KAN(bottleneck and decoder)的参数，并将linear encoder替换为Multi-head attention机制并fine-tuning. 优势：“This integration significantly improves the model’s ability to discern meaningful patterns amidst distribution shifts compared to standard positional encoding or no encoding.” ([Li 等, 2025, p. 11](zotero://select/library/items/2JTFKDQQ)) ([pdf](zotero://open-pdf/library/items/TVC6QHPY?page=11&annotation=GT8K2XUP))
    - 滑动窗口数据输入到Multi-head attention 中提取特征，然后使用KAN bottleneck压缩特征，最后通过KAN decoder重构输入数据
    - 压缩特征通过KNN计算并与控制阈值比较，判断是否存在fault，
      - 如果sample判断为故障，则不需要任何参数更新
      - 相反，如果判断为正常，则动态更新multi-head attention机制的参数(使用滑动窗口数据的损失)，以适应数据分布的变化。并且将

TTT的这块没有说清楚，什么是monitoring statistic被添加到sliding window里？
“In contrast, if the sample is recognized as normal, its monitoring statistic is added to a fixed-length sliding window (which functions as a dynamic collection of normal samples’ statistics, i.e., a memory buffer)” ([Li 等, 2025, p. 11](zotero://select/library/items/2JTFKDQQ)) ([pdf](zotero://open-pdf/library/items/TVC6QHPY?page=11&annotation=PKJXWQHK))
理解错了，真正的流程：
- N个时序样本，每个样本长为M，通过降维和KAN bottleneck提取特征为C维向量。
- 计算每个样本的KNN distance，即计算每个样本与其他样本的欧氏距离的均值，得到N个distance值。
- 通过percentile approach确定控制阈值
- 在test-time阶段，使用滑动窗口选取最近的L个样本（L<=N），计算这L个样本的KNN distance。
- 如果distance超过控制阈值，则判断为fault，否则为normal。
- 如果normal的话，通过loss更新encoder，并将normal样本添加到fixed-length sliding window中，同时删除oldest data，保持sliding window的长度不变。

