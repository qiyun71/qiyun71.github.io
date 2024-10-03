---
title: Dynamic Response Prediction
date: 2024-09-24 22:14:16
tags:
  - 
categories: Other Interest
---

FE Surrogate Model:
How to get dynamic response (time domain) through structural parameters?

<!-- more -->


# Machine Learning

## 


## [SM for DR](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10143077) 💧

Title: A Machine Learning-Based Surrogate Finite Element Model for Estimating Dynamic Response of Mechanical Systems

- Decision trees (DTs) and deep neural networks, 直接输入结构参数，输出time-domain series，精度不会很高
  - XGBoost Decision trees
  - AdaBoost Decision trees
  - RF: random fores
  - DNNs: deep NN


# LSTM

## [GNN-LSTM-based](https://www.sciencedirect.com/science/article/pii/S0141029624002955)

使用GNN从结构 graph中提取结构信息，与振动信息fusion后，输入LSTM进行预测 位移+速度+加速度序列数据

LSTM的输入包括：
- 图嵌入网络提取的结构信息
- ground-motion motion sequence，用了地震数据作为输入, Most of the ground-motion records used in the simulations did not contain [velocity pulses](https://www.sciencedirect.com/topics/engineering/pulse-velocity "Learn more about velocity pulses from ScienceDirect's AI-generated Topic Pages").

The LSTM model is thus capable of not only **retaining the long-term dependencies inherent in the ground-motion data** but also leveraging the structural features derived from graph embeddings.

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240924221607.png)
