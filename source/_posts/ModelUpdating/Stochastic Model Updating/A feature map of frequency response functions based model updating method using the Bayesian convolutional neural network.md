---
title: A feature map of frequency response functions based model updating method using the Bayesian convolutional neural network
date: 2024-05-26 14:01:17
tags:
  - 
categories: ModelUpdating/Stochastic Model Updating
---

| Title     | A feature map of frequency response functions based model updating method using the Bayesian convolutional neural network                                                                                                                                           |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Zhenyu Wang; Deli Liang; Sheng Ding; Wenliang Zhang; Huan He                                                                                                                                                                                                        |
| Conf/Jour | Mechanical Systems and Signal Processing                                                                                                                                                                                                                            |
| Year      | 2023                                                                                                                                                                                                                                                                |
| Project   | [A feature map of frequency response functions based model updating method using the Bayesian convolutional neural network - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327023007264?ref=pdf_download&fr=RR-2&rr=87df480b9ca904d1#s0030) |
| Paper     |                                                                                                                                                                                                                                                                     |



<!-- more -->

- [ ] the Rayleigh damping model，结构力学阻尼比假设(理论)
- [ ] [Bayesian convolutional neural network(**BCNN**)](https://arxiv.org/pdf/1901.02731)
- [x] 拉丁超立方采样(Latin hypercube sampling, LHS) 数据采样方法(离散思想)
- [x] Data preparation(features map双通道：实部和虚部)
- [x] 数据两步归一化方法(max_min & $\mu$,$\sigma$) 特征map数据处理方法
- [x] 所选参数的Correlation and sensitivity analysis
- [x] Analysis of the noise-resisting ability  往target data 中添加高斯白噪声


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240526140006.png)


## 归一化

### feature map 归一化

- $\hat{\mathscr{X}}^{(m)}=\frac{\mathscr{X}^{(m)}-x_{\min}}{x_{\max}-x_{\min}}\quad m=1,\cdots,M$
- $\stackrel{\smile}{\mathcal{X}}^{(m)}=\frac{\hat{\mathcal{X}}^{(m)}-\hat{\mu}}{\hat{\sigma}^2}\quad m=1,\cdots,M$

**不明白论文中为什么要分两步归一化，先min-max在mean-std与直接mean-std的结果是一样的**

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240527112135.png)


### 参数归一化
将每个单独的参数归一化为无量纲比例因子

|              | E1 (y1)         | E2 (y2) | E3 (y3) | E4 (y4)         | μ1 (y5)          | μ2 (y6) | μ3 (y7) | μ4 (y8)         | ζ1 (y9)        | ζ2 (y10) | ζ3 (y11)           |
| ------------ | --------------- | ------- | ------- | --------------- | ---------------- | ------- | ------- | --------------- | -------------- | -------- | ------------------ |
|              | Young’s modulus | of      |         | 4 substructures | Poisson’s ratios | of      |         | 4 substructures | Damping ratios | of       | the lowest 3 modes |
| Actual value | 204,000         | 185,000 | 164,800 | 123,600         | 0.28             | 0.24    | 0.26    | 0.31            | 0.028          | 0.035    | 0.056              |
| Scale factor | 0.9528          | 0.8077  | 0.6523  | 0.3354          | 0.50             | 0.10    | 0.30    | 0.80            | 0.300          | 0.417    | 0.767              |


## BCNN(Bayesian convolutional neural network)

> [Bayesian neural network introduction - 知乎](https://zhuanlan.zhihu.com/p/79715409)

可以看出构建的BCNN是根据一个双通道的feature map来预测对应的11个修正参数的值

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240526141041.png)



## Correlation and sensitivity analysis

1000 sets of model parameters $y_{1}\sim y_{11}$与feature maps of FRFs之间的相关性和敏感性分析：

相关性：$r=\frac{\sum_{i=0}^n(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=0}^n\left(x_i-\bar{x}\right)^2\sum_{i=0}^n\left(y_i-\bar{y}\right)^2}}$ (参数x与特征map之间的相关性计算, $r \in [-1,1]$)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240527105341.png)


敏感性：structural [similarity index](https://www.sciencedirect.com/topics/engineering/similarity-index "Learn more about similarity index from ScienceDirect's AI-generated Topic Pages") (SSIM) based and variance-based **Sobol** sensitivity analysis
- Sobol计算过程[敏感性分析—Sobol_sobol灵敏度分析-CSDN博客](https://blog.csdn.net/xiaosebi1111/article/details/46517409)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240527105331.png)

## Datasets

train data sets:
- 1000: randomly generated sets of data $\mathscr{D}^{(0)}$
- 3000: $\mathscr{D}^{(0)}$ + (1%, 3%, and 5%) gaussian noise

test data sets:
- 200 features maps of AFRFs with different parameters

## Result
### Accuracy of Network

训练结束后，randomly generated 200 feature maps of AFRFs of structures with different parameters to verify the accuracy of the network.

$f_{5}\sim f_{8}$效果不好的原因可能是泊松比与弹性模量重复设置，**泊松比对feature map 不敏感，且相关性很低**

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240527104905.png)


### Results of model updating


对比MC法与本文FMFRF方法的修正参数分布 (scale factors)
- BCNN只能估计正态分布...(红色的线)

![1-s2.0-S0888327023007264-gr16_lrg.jpg (3228×2840)](https://ars.els-cdn.com/content/image/1-s2.0-S0888327023007264-gr16_lrg.jpg)

两个表对比：
- Structural parameters | Experiment  | Initial FE model |  Error | Updated FE model | Error | Standard deviations
- Vibration characteristics | Experiment | Initial FE model | Error | Updated FE model | Error | Standard deviations

Distribution of natural frequencies (a-c) and modal damping ratios (d-f) predicted from the updated FE model.

![1-s2.0-S0888327023007264-gr17_lrg.jpg (2756×2456)](https://ars.els-cdn.com/content/image/1-s2.0-S0888327023007264-gr17_lrg.jpg)

### Analysis of the noise-resisting ability

3、5、7、10%
- 不同噪声水平下，FMFRF针对参数的修正结果，列表$y_{1}\sim y_{11}$
- 不同噪声水平下，FMFRF修正后的AFRF与试验的差别

![1-s2.0-S0888327023007264-gr18_lrg.jpg (3169×1455)](https://ars.els-cdn.com/content/image/1-s2.0-S0888327023007264-gr18_lrg.jpg)

![1-s2.0-S0888327023007264-gr22_lrg.jpg (3150×1479)](https://ars.els-cdn.com/content/image/1-s2.0-S0888327023007264-gr22_lrg.jpg)

**为什么能降噪**：
- the **BCNN itself** has good **noise resistance**
- the training set of BCNN includes **training samples composed of AFRFs with different noise levels**, which results in the reduction of the error margin of the updated results,
- the **integration process,** as shown in Eq. [(12)](https://www.sciencedirect.com/science/article/pii/S0888327023007264?ref=pdf_download&fr=RR-2&rr=87df480b9ca904d1#e0060), to some extent “neutralizes” the noise components of AFRFs during the feature map generation process.

Eq.12: **构建FRF map的好想法**

实部FRF：$\begin{aligned}x_{i_1,i_2,i_3}&=\frac{v}{\omega_{\max}-\omega_{\min}}\int_{\omega_{\min}+\frac{\omega_{\max}-\omega_{\min}}{v}{(i_3-1)}}^{\omega_{\min}+\frac{\omega_{\min}}{v}{i_3}}P_{p_{i_2},q}^{\mathrm{a}}(\omega,y_1,\cdots,y_o)\mathrm{d}\omega\\i_1&=1i_2=1,2,\cdots,ui_3=1,2,\cdots,v\end{aligned}$
虚部FRF：$\begin{aligned}x_{i_1,i_2,i_3}&=\frac{v}{\omega_{\max}-\omega_{\min}}\int_{\omega_{\min}+\frac{\omega_{\max}-\omega_{\min}}{v}{(i_3-1)}}^{\omega_{\min}+\frac{\omega_{\max}-\omega_{\min}}{v}{i_3}}Q_{p_{i_2},q}^{\mathrm{a}}(\omega,y_1,\cdots,y_o)\mathrm{d}\omega\\i_1&=2i_2=1,2,\cdots,ui_3=1,2,\cdots,v\end{aligned}$


# Future

在未来的工作中，
- 本文的方法将应用于**实际的工程结构**。
- 此外，需要指出的是，FRF的特征图中没有考虑测点的空间信息，原则上，测点的空间位置也具有重要的现实意义。**空间位置信息**可以扩展到未来FRF的特征图中。

空间位置信息：
- 单纯在输入的时候添加空间位置的坐标信息？
- 结合NeRForOthers