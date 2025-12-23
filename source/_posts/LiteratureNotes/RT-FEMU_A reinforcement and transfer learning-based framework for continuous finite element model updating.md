---
zotero-key: 5T36MBG7
zt-attachments:
  - "7943"
title: "RT-FEMU: A reinforcement and transfer learning-based framework for continuous finite element model updating"
created: 2025-10-31 07:55:10
modified: 2025-10-31 07:57:29
tags: ⭐⭐⭐  Structural damage identification  Transfer learning  Structural health monitoring (SHM)  Finite element model updating (FEMU)  Reinforcement learning (RL)
collections: Inverse SM
year: 2025
publication: Journal of Building Engineering
citekey: wangRTFEMUReinforcementTransfer2025
author:
  - Zhen Wang
  - Yuqing Gao
---
| Title        | "RT-FEMU: A reinforcement and transfer learning-based framework for continuous finite element model updating"                                                                                                                                                                                   |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author       | [Zhen Wang,Yuqing Gao]                                                                                                                                                                                                                                                                          |
| Organization |                                                                                                                                                                                                                                                                                                 |
| Paper        | [Zotero pdf](zotero://select/library/items/5T36MBG7) [attachment](<file:///D:/Download/Zotero_data/storage/JQ7Y62DL/Wang%E5%92%8CGao%20-%202025%20-%20RT-FEMU%20A%20reinforcement%20and%20transfer%20learning-based%20framework%20for%20continuous%20finite%20element%20model%20u.pdf>)<br><br> |
| Project      |                                                                                                                                                                                                                                                                                                 |

<!-- more -->

## Background

Accurate finite element models are essential
However, due to construction imperfections, environmental degradation, and extreme events, 实际结构和FEM之间存在差异

## Innovation

“a continuous finite element model updating (FEMU) framework, termed RT-FEMU” ([Wang和Gao, 2025, p. 1](zotero://select/library/items/5T36MBG7)) ([pdf](zotero://open-pdf/library/items/JQ7Y62DL?page=2&annotation=3WEWUVZC)) 将FEMU重构为tracking problem(a customized reward function, a defined termination criterion, and advanced RL algorithms)
- Reinforcement learning
- Transfer learning

## Outlook

## Cases

## Equation

迁移学习：

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251031160956.png)

Overview：
- “due to construction imperfections, environmental degradation, and extreme events” ([Wang和Gao, 2025, p. 1](zotero://select/library/items/5T36MBG7)) ([pdf](zotero://open-pdf/library/items/JQ7Y62DL?page=2&annotation=LBYS5UHP)) 测量与仿真之间总有差异
- FEMU-Agent：与Env交互不断调整学习optimal update policy
  - 使用DNN来近似policy function
  - 使用Proximal Policy Optimization (PPO)算法来训练DNN
  - policy function by actor network：$\pi(a|s;\theta)$ , 表示Agent选择action的policy
  - value function by critic network：$V(s;\omega )$，表示当前状态的value estimation
- FEMU-Env由SHM系统、实体结构（动态变化的测量响应 ）和FEM组成(待refine的FEM)
  - 通过SHM系统测量实体结构的响应
  - 推荐使用参数化的FEM，方便steamlined modification

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251031171122.png)

- Sate：实际结构的测量响应与FEM的仿真响应
  - raw measured response通过filtering pipeline，并与FEM predicted esponse 张量编码到一起。(虽然结合时频信息，或者其他结构信息等可以增强observation space，但是本文只用了频域表示)
  - 由于外部激励无测量数据、结构相应噪声和高维时序的计算量大的问题，不直接使用测量的时域数据，而是通过modal identification(频域Frequency Domain Decomposition、时域Stochastic Subspace Identification)从filtered 频域响应中提取关键信息(固有频率、模态振型和阻尼比)。最终辨识的modal parameters from 试验(measured)和FEM（simulated）被编码到一起作为当前观察的state。

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251101155634.png)


- Action
  - 通过工程经验大致确定哪些FEM参数对响应有影响，然后通过敏感性分析选择最敏感的几个参数，最后使用组合搜索方法确定每个参数的调整范围
  - action space
    - 连续action space：model parameter的修改范围
    - 离散action space：每个参数的修改范围被离散化为几个预定义的值
    - normalized action space：将连续action space归一化到[0,,1]区间

- Reward
	- natural frequencies：L2
	- mode shapes：Modal Assurance Criterion (MAC) and L2 and modal shape curvature


基于迁移学习的连续更新机理：

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20251103164208.png)
