---
title: A novel interval model updating framework based on correlation propagation and matrix-similarity method
date: 2024-03-12 16:52:21
tags:
  - 
categories: ModelUpdating/Interval Model Updating
---

| Title     | A novel interval model updating framework based on correlation propagation and matrix-similarity method                                                                                                 |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Baopeng Liao , Rui Zhao , Kaiping Yu , Chaoran Liu 哈工大                                                                                                                                                  |
| Conf/Jour | Mechanical Systems and Signal Processing                                                                                                                                                                |
| Year      | 2022                                                                                                                                                                                                    |
| Project   | https://www.sciencedirect.com/science/article/abs/pii/S0888327021004313                                                                                                                                 |
| Paper     | [A novel interval model updating framework based on correlation propagation and matrix-similarity method](https://readpaper.com/pdf-annotate/note?pdfId=2202716203238606336&noteId=2202716355810672896) |

<!-- more -->

模型更新技术在实际系统中具有不确定性的数值模型中得到了广泛的应用，而随机理论在知识不足的情况下是无效的。此外，面对相关的不确定性和复杂的数值模型，模型更新仍然是一个挑战。本文提出了一种新的区间模型更新框架，以解决极限样本的相关不确定性问题。这种框架的优点是，**无论输入输出关系是线性的还是非线性的**，都可以高精度地更新参数。为了实现这一优势，**采用凸建模技术和Chebyshev代理模型**分别进行不确定参数量化和数值模型逼近。随后，**提出了考虑关联传播的矩阵相似法**，构建了两步区间模型更新过程，将其转化为确定性模型更新问题。精神的结果。值得注意的是，三个例子验证了所提出的框架在线性和非线性关系中的有效性和优越性。结果表明，本文提出的区间模型更新框架适用于处理参数边界及其相关性的更新问题。

算例：两个数值算例和一个实验算例验证
- **a classical mass-spring system**
- the composite beam structure
- the real physical system through the experiments of beam structure with sliding masses 带滑动质量的梁结构