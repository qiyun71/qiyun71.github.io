---
zotero-key: RJAE4SPC
zt-attachments:
  - "5314"
title: Deep learning for high-dimensional reliability analysis
created: 2025-08-18 07:36:16
modified: 2025-08-18 07:36:17
tags:
  - Deep
  - learning
  - Reliability
  - /done
  - Dimension
  - reduction
  - Uncertainty
  - quantification
  - Gaussian
  - process
  - Autoencoder
collections: Reliability
year: 2020
publication: Mechanical Systems and Signal Processing
citekey: liDeepLearningHighdimensional2020
---

| Title        | "Deep learning for high-dimensional reliability analysis"                                                                                          |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author       | [Mingyang Li,Zequn Wang]                                                                                                                           |
| Organization | Department of Mechanical Engineering-Engineering Mechanics, Michigan Technological University, Houghton, MI 49931, **USA**                         |
| Paper        | [Zotero pdf](zotero://select/library/items/RJAE4SPC) [attachment](file:///D:/Download/Zotero_data/storage/4YCSUPIT/S088832701930620X.html)<br><br> |
| Project      |                                                                                                                                                    |

<!-- more -->

## Background

高维可靠性分析面临curse of dimensionality

## Innovation

“a novel highdimensional data abstraction (HDDA) framework” ([Li和Wang, 2020, p. 1](zotero://select/library/items/RJAE4SPC)) ([pdf](zotero://open-pdf/library/items/ZWMFIZ6K?page=1&annotation=TI5TQ7X3))
- 训练autoencoder将 input X 和ouput Y 降维到latent variables $\theta$ in latent space
- 训练deep feedforward neural network构建 input X 到 latent variables  $\theta$ 的映射
- 训练gaussian process model 构建 latent variables  $\theta$ 到 output Y 的映射
- adaptive sampling(only reliability)

相比于简单的先训练 X -->  $\theta_{X}$, Y -->  $\theta_{Y}$ autoencoder，然后训练 X -->  $\theta_{X}$ --> surrogate model --> $\theta_{Y}$ -->  Y。训练从X,Y到$\theta$，latent variables可以包含更多的信息。

## Outlook

- 只考虑了高维的输入，没有考虑高维的输出
- autoencoder 训练结束后，只是用来生成X 到 latent variables的样本。最终的surrogate model是DFNN与Kriging的结合。

## Cases

## Equation

![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250818154923.png)

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20250818162910.png)
