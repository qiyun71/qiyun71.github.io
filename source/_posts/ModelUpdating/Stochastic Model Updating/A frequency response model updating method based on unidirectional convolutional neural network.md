---
title: A frequency response model updating method based on unidirectional convolutional neural network
date: 2024-03-12 16:38:10
tags:
  - 
categories: ModelUpdating/Stochastic Model Updating
---

| Title     | A frequency response model updating method based on unidirectional convolutional neural network                                                                                                                 |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Xinjie Zhang , Zhongmin Deng, and Yanlin Zhao                                                                                                                                                                   |
| Conf/Jour | Mechanics of Advanced Materials and Structures                                                                                                                                                                  |
| Year      | 2019                                                                                                                                                                                                            |
| Project   | https://www.tandfonline.com/doi/abs/10.1080/15376494.2019.1681037                                                                                                                                               |
| Paper     | [A frequency response model updating method based on unidirectional convolutional neural network (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=2110304823896008192&noteId=2110305055136380672) |

<!-- more -->


本文提出了一种基于**单向卷积神经网络** (UCNN) 的方法，以利用频率响应 (FR) 数据进行有限元模型更新。UCNN 旨在在没有任何人工特征提取的情况下从 FR 数据获取高精度逆映射到更新参数。单向卷积分别应用于 FR 数据的频率和位置维度，以避免数据耦合。UCNN 在**卫星模型更新**实验中优于基于残差的模型更新方法和二维卷积神经网络。它在训练集中内外都实现了高精度的结果。

评价指标：
- FR 保证准则(assurance criterion) $FRAC=\frac{|H_e^TH_s|^2}{(H_e^TH_e)(H_s^TH_s)}$ 用于描述试验数据与模拟 FR 数据的相似度

算例：
- 卫星算例1