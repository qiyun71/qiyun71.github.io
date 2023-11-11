---
title: DiffuStereo
date: 2023-11-11 12:07:22
tags:
  - Diffusion
  - 3DReconstruction
categories: HumanBodyReconstruction/Generative approach
---

| Title     | DiffuStereo: High Quality Human Reconstruction via Diffusion-based Stereo Using Sparse Cameras |
| --------- | ---------------------------------------------------------------------------------------------- |
| Author    | Ruizhi Shao, Zerong Zheng, Hongwen Zhang, Jingxiang Sun, Yebin Liu                             |
| Conf/Jour | ECCV 2022 Oral                                                                                 |
| Year      | 2022                                                                                           |
| Project   | [DiffuStereo Project Page (liuyebin.com)](https://liuyebin.com/diffustereo/diffustereo.html)   |
| Paper     | [DiffuStereo: High Quality Human Reconstruction via Diffusion-based Stereo Using Sparse Cameras (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4648031844416110593&noteId=2044584813117342208)                                                                                               |

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231111120652.png)

<!-- more -->

# Abstract

我们提出了 DiffuStereo，这是一种仅使用稀疏相机(本工作中有 8 个)进行高质量 3D 人体重建的新系统。其核心是一种新型的基于扩散的立体模型，将扩散模型这一功能强大的生成模型引入到迭代立体匹配网络中。为此，我们设计了一个新的扩散核和附加的立体约束来促进网络中的立体匹配和深度估计。我们进一步提出了一个多级立体网络架构来处理高分辨率(高达 4k)的输入，而不需要负担得起的内存占用。给定一组稀疏视图彩色人体图像，基于多层次扩散的立体网络可以生成高精度的深度图，然后通过高效的多视图融合策略将深度图转换为高质量的三维人体模型。总的来说，我们的方法可以自动重建人体模型，其质量与高端密集视角相机平台相当，这是使用更轻的硬件设置实现的。实验表明，我们的方法在定性和定量上都大大优于最先进的方法。

## Introduction

贡献:
1)我们提出了 DiffuStereo，这是一个轻量级和高质量的系统，用于稀疏多视图相机下的人体体积重建。
2)据我们所知，我们提出了第一个将扩散模型引入立体和人体重建的方法。我们通过精心设计一个新的扩散核并在扩散条件中引入额外的立体约束来扩展香草扩散模型。
3)我们提出了一种新的多层次扩散立体网络，以实现准确和高质量的人体深度估计。我们的网络可以优雅地处理高分辨率(高达 4k)图像，而不会遭受内存过载的困扰。

# Method

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231111120652.png)

DiffuStereo 可以从稀疏的(少至 8 个)摄像机中重建高质量的人体模型（所有摄像机均匀分布在目标人周围的一个环上）。这种稀疏的设置给高质量的重建带来了巨大的挑战，因为两个相邻视图之间的角度可以大到 45 度。DiffuStereo 通过一种现成的重建方法 DoubleField[52]、一种基于扩散的立体网络和一种轻量级多视图融合模块的共同努力，解决了这些挑战。

DiffuStereo 系统由三个关键步骤组成:
- 通过 DoubleField[52]预测初始人体网格，并渲染为粗视差流(第 3.1 节); 作者先前提出的 DoubleField，最新的最先进的人体重建方法之一，表面和辐射场被桥接以利用人体几何先验和多视图外观，在给定稀疏视图输入的情况下提供良好的网格初始化。
- 对于每两个相邻视图，在基于扩散的立体中对粗视差图进行细化，得到高质量的深度图(第 3.2 节);基于扩散的立体网络对每个输入视图的视差图有很强的改进能力，其中使用扩散过程进行连续的视差细化。
- 初始的人体网格和高质量的深度图被融合成最终的高质量人体网格(第 3.3 节)，其中一个轻量级的多视图融合模块以初始网格作为锚点位置，有效地组装了部分精细的深度图。

## Mesh, Depth, Disparity Initialization

DoubleField 得到初始人体 mesh，然后渲染为深度图

m 和 n 是两个相邻视图的索引，为了得到视图 m 到相邻视图 n 的粗视差图 $x_{c}$，取视图 m 的深度图 $D^m_c$，计算像素位置 o = (i, j)处的视差: $\mathbf{x}_c(\boldsymbol{o})=\pi^n\left((\pi^m)^{-1}\left([\boldsymbol{o},\mathbf{D}_c^m(\boldsymbol{o})]^\mathrm{T}\right)\right)-\boldsymbol{o}$

其中 $(\pi^{m})^{-1}$ 将深度图 $D^m_c$ 中的点变换为世界坐标系，$π^n$ 将世界坐标系中的点投影为图像坐标系。

由于初始视差图是从粗糙的人体网格中计算的，因此可以在很大程度上缓解大位移和遮挡区域的问题。正如即将介绍的那样，**这些视差图通过 Diffusion-based Stereo 进一步细化，以获得每个输入视点的高质量深度图**

## Diffusion-based Stereo for Disparity Refinement

现有的(stereo methods)立体方法 [63MVSNet: Depth Inference for Unstructured Multi-view Stereo (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4518062699161739265&noteId=1986540055632613120) [9、20、23、68、33、63、69、70]采用3D/4D 成本体离散预测视差图，难以实现亚像素级的流量 flow 估计。为了克服这一限制，我们提出了一种基于扩散的立体网络，使立体网络可以在迭代过程中学习连续流。
具体来说，我们的基于扩散的立体图像包含一个正演过程和一个反演过程，以获得最终的高质量视差图。在正演过程中，将初始视差图扩散到有噪声分布的图中。在相反的过程中，在具有多个立体相关特征的条件下，从噪声图中恢复出高质量的视差图。
下面，我们简要介绍了一般的扩散模型，然后介绍了我们将扩散模型的连续性与基于学习的迭代立体相结合的解决方案。此外，我们还提出了一个多层次的网络结构，以解决高分辨率图像输入的内存问题。

### Generic Diffusion Model

More formally, the diffusion model can be written as two Markov Chains:
$$
\begin{aligned}
&q(\mathbf{y}_{1:T}|\mathbf{y}_{0}) =\prod\limits_{t=1}^Tq(\mathbf{y}_t|\mathbf{y}_{t-1}),  \\
&q(\mathbf{y}_t|\mathbf{y}_{t-1}) =\mathcal{N}(\sqrt{1-\beta_t}\mathbf{y}_{t-1},\beta_tI)  \\
&p_{\theta}(\mathbf{y}_{0:T}|\mathbf{s}) =p(\mathbf{y}_T)\prod_{t=1}^Tp_\theta(\mathbf{y}_{t-1}|\mathbf{y}_t,\mathbf{s}), 
\end{aligned}
$$

其中$q(\mathbf{y}_{1:T}|\mathbf{y}_{0})$为正向函数，$q(\mathbf{y}_t|\mathbf{y}_{t-1})$为扩散核，表示加入噪声的方式，$\mathcal{N}$为正态分布，h the identical matrix，pθ()为反向函数，采用去噪网络Fθ对yt进行去噪，为附加条件。当T→∞时，正反过程可以看作是连续过程或随机微分方程[55]，这是连续流量估计的自然形式。正如之前的工作[55]所验证的那样，在参数更新中注入高斯噪声使迭代过程更加连续，并且可以避免陷入局部极小。在这项工作中，我们将展示，这样一个强大的生成工具也可以用来为以人为中心的立体任务产生连续的流。