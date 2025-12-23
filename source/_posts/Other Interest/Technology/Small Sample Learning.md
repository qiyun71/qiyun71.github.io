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

Machine Learning： Learning Algorithm $\mathcal{A}$ helps Learners $\mathcal{F}$ improve certain performance measure $\mathcal{P}$ when executing some tasks, through precollected experiential Data $\mathcal{D}$. 通常需要大量的 labeled data

SSL: 各种方法的分类
- Experience learning： 
  - Augmented data：compensate the input data with other sources of data
  - Knowledge system:
    - Representations from other domains (Transfer Learning)
    - Trained models (Fine-tuing)
    - Cognition knowledge on concepts: Such knowledge include **common sense knowledge**, **domain knowledge**, and **other prior knowledge** on the learned concept with small training samples. 例如想要识别眼睛的位置，可以告诉网络其在鼻子/嘴巴上方
    - Meta knowledge：Some high-level knowledge beyond data. 例如 1+1=2 的 meta knowledge 就是告诉网络：1、2 的意义，+、=的计算模式
- Concept learning： aims to perform recognition or form new concepts (samples) from few observations (samples) through fast processing (employs **matching rule** $\mathcal{R}$ to associate concepts in **concept system** $\mathcal{C}$ with input small samples $\mathcal{S}$)
  - Concept system
    - Intensional representations of concept: precise definitions in proposition or semantic form on the learned concept, like its **attribute characteristics** 就是物体的属性，比如斑马的颜色是黑色+白色，而不是棕色 or others
    - Extension representations of concept: prototypes and instances related to the learned concept. 是物体的原型/实例，如斑马的照片
  - Matching rule: a procedure to associate concepts in concept system C with small samples S to implement a cognition or recognition task. The result tries to keep optimal in terms of performance measure P.

k-shot learning: （just describes a setting manner of the SSL problem）[详细介绍](https://arxiv.org/pdf/1808.04572#page=9.11)

# Method

## LOOCV

> [(10 封私信) 机器学习如何在小样本高维特征问题下获得良好表现？ - 知乎](https://www.zhihu.com/question/264240892)

[LOOCV - Leave-One-Out-Cross-Validation 留一交叉验证-CSDN博客](https://blog.csdn.net/dpengwang/article/details/84934197)

正常训练都会划分训练集和验证集，训练集用来训练模型，而验证集用来评估模型的泛化能力。留一交叉验证是一个极端的例子，如果数据集 D 的大小为 N，那么用 N-1 条数据进行训练，用剩下的一条数据作为验证，用一条数据作为验证的坏处就是可能 $E_{val}$ 和 $E_{out}$ 相差很大，
所以**在留一交叉验证里**，每次从 D 中取一组作为验证集，直到所有样本都作过验证集，共计算 N 次，最后对验证误差求平均，得到 Eloocv(H,A),这种方法称之为留一法交叉验证

## Siamese NNs

[卷积神经网络学习笔记——Siamese networks（孪生神经网络） - 战争热诚 - 博客园](https://www.cnblogs.com/wj-1314/p/11556107.html)

通过两个相同的网络来对比学习同一类物体

![1226410-20200427155758079-2086836326.png (692×489)|444](https://img2020.cnblogs.com/blog/1226410/202004/1226410-20200427155758079-2086836326.png)

# Dimension reduction

[Data Analysis](../../Learn/Python/Data%20Analysis.md#数据降维)
