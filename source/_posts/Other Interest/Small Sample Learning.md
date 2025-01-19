---
title: Small Sample Learning
date: 2024-12-27 19:18:15
tags:
  - 
categories: Other Interest
---

小子样学习

<!-- more -->

# Basic

# Paper

>  [Small Sample Learning in Big Data Era](https://arxiv.org/pdf/1808.04572) Review

Machine Learning： Learning Algorithm $\mathcal{A}$ helps Learners $\mathcal{F}$ improve certain performance measure $\mathcal{P}$ when executing some tasks, through precollected experiential Data $\mathcal{D}$. 通常需要大量的labeled data

SSL: 各种方法的分类
- Experience learning： 
  - augmented data：compensate the input data with other sources of data
  - knowledge system:
    - Representations from other domains (Transfer Learning)
    - Trained models (Fine-tuing)
    - Cognition knowledge on concepts: Such knowledge include **common sense knowledge**, **domain knowledge**, and **other prior knowledge** on the learned concept with small training samples. 例如想要识别眼睛的位置，可以告诉网络其在鼻子/嘴巴上方
    - Meta knowledge：Some high-level knowledge beyond data. 例如1+1=2的meta knowledge就是告诉网络：1、2的意义，+、=的计算模式
- Concept learning： aims to perform recognition or form new concepts (samples) from few observations (samples) through fast processing (employs **matching rule** $\mathcal{R}$ to associate concepts in **concept system** $\mathcal{C}$ with input small samples $\mathcal{S}$)
  - Concept system
    - intensional representations of concept: precise definitions in proposition or semantic form on the learned concept, like its **attribute characteristics** 就是物体的属性，比如斑马的颜色是黑色+白色，而不是棕色or others
    - extension representations of concept: prototypes and instances related to the learned concept. 是物体的原型/实例，如斑马的照片
  - Matching rule: a procedure to associate concepts in concept system C with small samples S to implement a cognition or recognition task. The result tries to keep optimal in terms of performance measure P.

k-shot learning: （just describes a setting manner of the SSL problem）[详细介绍](https://arxiv.org/pdf/1808.04572#page=9.11)

