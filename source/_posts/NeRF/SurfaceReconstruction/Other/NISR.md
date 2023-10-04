---
title: 未命名
date: 2023-09-26 14:20:04
tags:
  - Loss
categories: NeRF/SurfaceReconstruction/Other
---

| Title     | Improving Neural Indoor Surface Reconstruction with Mask-Guided Adaptive Consistency Constraints                                                                                                                 |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Xinyi Yu1, Liqin Lu1, Jintao Rong1, Guangkai Xu2,∗ and Linlin Ou1                                                                                                                                                |
| Conf/Jour |                                                                                                                                                                                                                  |
| Year      | 2023                                                                                                                                                                                                                 |
| Project   |                                                                                                                                                                                                                  |
| Paper     | [Improving Neural Indoor Surface Reconstruction with Mask-Guided Adaptive Consistency Constraints (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4801757208966594561&noteId=1976543992973611776) |

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230925133140.png)

Idea:
- 法向预测网络，法向量约束
- 一致性约束(几何一致性和颜色一致性)，通过虚拟视点实现
- mask的计算方法，只计算**有价值**的光线
<!-- more -->

# Abstract
从2D图像中重建3D场景一直是一个长期存在的任务。最近的研究不是估计每帧深度图并在3D中融合它们，而是利用神经隐式表面作为3D重建的统一表示。配备数据驱动的预训练几何线索，这些方法已经证明了良好的性能。然而，不准确的先验估计通常是不可避免的，这可能导致重建质量次优，特别是在一些几何复杂的区域。在本文中，我们提出了一个两阶段的训练过程，解耦视图相关和视图无关的颜色，并利用**两个新的一致性约束**来增强细节重建性能，而**不需要额外的先验**。此外，我们引入了一个基本掩码方案来自适应地影响监督约束的选择，从而提高自监督范式的性能。在合成数据集和真实数据集上的实验表明，该方法能够减少先验估计误差的干扰，实现具有丰富几何细节的高质量场景重建。

# Method
穿过一张图片$I_{k}$采样光线，并随机生成对应的虚拟光线，然后NeRF MLP渲染颜色、视图独立颜色、深度和法向。通预训练的法向估计模型来估计光线对应像素的法向，然后最小化颜色损失来优化MLP，此外还添加了mask驱动的一致性约束和法向约束

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230925133716.png)

使用两个颜色网络来训练：与视图有关的颜色$\hat c^{vd}$和与视图无关的颜色$\hat c^{vi}$
深度图和法向量图的计算：$\hat{D}(r)=\sum_{i=1}^nT_i\alpha_it_i,\quad\hat{\mathbf{N}}(\mathbf{r})=\sum_{i=1}^nT_i\alpha_i\hat{\mathbf{n}}_i$

监督约束：
- 颜色：$\mathcal{L}_{rgb}=\sum_{\mathbf{r}\in\mathcal{R}}\left\|\hat{\mathbf{C}}(\mathbf{r})-\mathbf{C}(\mathbf{r})\right\|_1$
- 法向量约束$\begin{aligned}\mathcal{L}_{normal}&=\frac1{\left|\mathcal{M}_r\right|}\sum_{\text{r}\in\mathcal{M}_r}\left\|\hat{\mathbf{N}}(\mathbf{r})-\bar{\mathbf{N}}(\mathbf{r})\right\|_1\\&+\left\|1-\hat{\mathbf{N}}(\mathbf{r})^T\bar{\mathbf{N}}(\mathbf{r})\right\|_1\end{aligned}$ ，其中$M_{r}$为射线掩模
- 几何一致性约束
    - 通过采样像素的射线生成深度图，根据深度图计算出目标3D点的位置。然后随机生成一个虚拟视点 ，根据3D目标点位置和虚拟视点可以计算出虚拟射线的方向$\mathbf{x}_t=\mathbf{o}+\hat{D}(\mathbf{r})\mathbf{v},\quad\mathbf{v}^v=\frac{\mathbf{x}_t-\mathbf{o}^v}{\left\|\mathbf{x}_t-\mathbf{o}^v\right\|_2}$
    - 根据虚拟射线的视点和方向，可以由渲染框架MLP得到虚拟射线的深度图$\hat{D}(\mathbf{r}_{v})$和法向量图$\hat{\mathrm{N}}(\mathbf{r}_v)$，由两光线深度的几何一致性$\mathcal{L}_{gc}=\frac{1}{2|\mathcal{M}_{v}|}\sum_{\mathbf{r}_{v}\in\mathcal{M}_{v}}|\hat{D}(\mathbf{r}_{v})-\bar{D}(\mathbf{r}_{v})|^{2}$，其中$\bar{D}(\mathbf{r}_v)=\left\|\mathbf{x}_t-\mathbf{o}^v\right\|_2$

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230925140953.png)


- 光度一致性约束
    - 两光线渲染得到的像素颜色：$\mathcal{L}_{pc}=\frac{1}{|\mathcal{M}_{r}|}\sum_{\mathbf{r}\in\mathcal{M}_{r}}\left\|\hat{\mathbf{C}}_{vi}(\mathbf{r})-\hat{\mathbf{C}}_{vi}(\mathbf{r}_{v})\right\|_{1}$

$\mathcal{M}_{r}$与$\mathcal{M}_{v}$选择方法：
- Sample Mask：保证生成虚拟视点在物体外部
    - $\left.\mathcal{M}_s=\left\{\begin{array}{lr}1,&\quad if\hat{s}(\mathbf{o}_v)>0.\\0,&\quad otherwise.\end{array}\right.\right.$
- Occlusion Mask：解决由于两条光线的遮挡导致的深度一致性误差问题（两条光线都只穿过物体一次）
    - $\left.\mathcal{M}_o^s=\left\{\begin{array}{lr}1,&if~\|diff(sgn(\hat{\mathbf{s}}))\|_1\leq2.\\0,&otherwise.\end{array}\right.\right.$
    - $\left.\mathcal{M}_o^v=\left\{\begin{array}{ccc}1,&if&\|diff(sgn(\hat{\mathbf{s}}^v))\|_1\leq2.\\0,&&otherwise.\end{array}\right.\right.$
    - $\mathcal{M}_o=\mathcal{M}_o^s\&\mathcal{M}_o^v$
- Adaptive Check Mask
    - 两个光线得到像素的法向量值夹角余弦，与某个阈值比较
    - $cos(\hat{\mathbf{N}}(\mathbf{r}),\hat{\mathbf{N}}(\mathbf{r}_v))=\frac{\hat{\mathbf{N}}(\mathbf{r})\cdot\hat{\mathbf{N}}(\mathbf{r}_v)}{\left\|\hat{\mathbf{N}}(\mathbf{r})\right\|_2\left\|\hat{\mathbf{N}}(\mathbf{r}_v)\right\|_2}$
    - $\left.\mathcal{M}_a=\left\{\begin{array}{lr}1,\quad&ifcos(\hat{\mathbf{N}}(\mathbf{r}),\hat{\mathbf{N}}(\mathbf{r}_v))<\epsilon.\\0,&otherwise.\end{array}\right.\right.$
- Mask integration
    - 法向量约束中mask $\mathcal{M}_{r}$：$\mathcal{M}_r=\mathcal{M}_s\&\mathcal{M}_o\&(1-\mathcal{M}_a)$
        - **两个光线得到像素的法向量值夹角余弦**大于某个阈值，无虚拟视点
    - 几何一致性约束中mask $\mathcal{M}_{v}$：$\mathcal{M}_v=\mathcal{M}_s\&\mathcal{M}_o\&\mathcal{M}_a$
        - **两个光线得到像素的法向量值夹角余弦**小于某个阈值，有虚拟视点

# Experiments

一个NVIDIA RTX 3090 GPU
- ScanNet数据集
- 对比COLMAP、NeuralRecon、MonoSDF（MLP Version）、NeuRIS
- 指标：Accuracy、Completeness、Chamfer Distance、Precision、Recall、F-score、Normal Consistency
