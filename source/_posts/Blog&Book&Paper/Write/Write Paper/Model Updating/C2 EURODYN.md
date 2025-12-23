---
title: C2
date: 2025-10-14 13:20:35
tags: 
categories: Blog&Book&Paper/Write/Write Paper/Model Updating
Year: 
Journal:
---

| Title        | C2 |
| ------------ | ------------------- |
| Author       |                     |
| Organization |                     |
| Paper        |                     |
| Project      |                     |

<!-- more -->


| **Conference Paper** | ⭐   | [Call for Papers — EURODYN 2026](https://eurodyn2026.org/call-for-papers/) | 2025.10.25 | [20251012_ConferencePaper_abstract_for_EURODYN2026](file:///D:%5CDownload%5CBaiduSyncdisk%5CReport%5C20251012_ConferencePaper_abstract_for_EURODYN2026)<br>[C2 EURODYN](Write/Write%20Paper/Model%20Updating/C2%20EURODYN.md) | dynamic+生成式AI in MU/SHM，or NSFC中找一个合适主题 |
| -------------------- | --- | -------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |

研究内容：

高精度：                             
~~传播机理研究~~
~~多场不确定性传播 DA-MCMC~~
~~敏感性分析~~

高效率：
多保真自适应建模 高中低
多保真度智能切换和协同验证

高可靠性：
~~风险置信度量化~~ ~~基于贝叶斯推断~~
~~可靠性优化RBDO~~ ~~基于不确定性~~

小子样：
动态实时模型修正 cINN
主动学习策略 试验数据与仿真模型交互


Virtual sample generation based on variational autoencoders for model updating of dynamical system under small sample conditions

Numerical simulation is indispensable for the analysis and design of complex engineering systems, particularly in the field of structural dynamics. To enhance the predictive accuracy of these simulations, model updating techniques are employed to align numerical models with experimental observations. However, the efficacy of many data-driven updating methods is severely hampered when physical experiments or high-fidelity simulations are expensive or time-consuming, leading to a critical challenge of small sample sizes. This paper proposes a novel model updating framework that leverages variational autoencoders (VAEs) for virtual sample generation to overcome this data scarcity issue. The proposed method first trains two VAEs on the limited available data to learn the underlying low-dimensional latent representation and probability distribution of structural parameters and system responses. The Kullback-Leibler (KL) divergence loss is utilized to regularize the learned latent distributions towards same standard normal distribution, thereby facilitating efficient sampling during the inference phase and capturing the intrinsic correlation between parameters and responses. Subsequently, the trained VAEs are utilized as a generative model to produce a large number of high-quality virtual samples that faithfully augment the original dataset. These augmented data are then used to construct a high-fidelity surrogate model, which can efficiently and accurately map model parameters to system responses, thus facilitating a robust updating process. The efficacy and robustness of the proposed approach are demonstrated through two numerical case studies on benchmark dynamical systems. The results indicate that the VAE-based sample generation strategy significantly improves the accuracy of the updated model compared to traditional methods, especially under conditions of severe data limitation.

>  ~~The Kullback-Leibler (KL) divergence loss is utilized to minimize the difference of learned latent distributions between the parameters and responses, thereby capturing the intrinsic correlation between them.~~
>  这样可以捕捉参数和响应之间的内在相关性，但是如果不约束到标准正态分布，在inference时很难采样 


大致思路：
[overview.pptx](file:///D:%5CDownload%5CBaiduSyncdisk%5CReport%5CP4_GSA+AK%20model+SIS_SCI%5CFigure%5COverview%5Coverview.pptx)