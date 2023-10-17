---
title: PIFu
date: 2023-09-28 17:44:42
tags:
  - PIFu
categories: HumanBodyReconstruction/ImplicitFunction
---

| Title     | PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization                                                                                                                 |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Shunsuke Saito1,2 * Zeng Huang1,2 * Ryota Natsume3 * Shigeo Morishima3 Angjoo Kanazawa4Hao Li1,2,5                                                                                                   |
| Conf/Jour | ICCV                                                                                                                                                                                                 |
| Year      | 2019                                                                                                                                                                                                 |
| Project   | [PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization (shunsukesaito.github.io)](https://shunsukesaito.github.io/PIFu/)                                              |
| Paper     | [PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4518249702759227393&noteId=1981090816765700608) |

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230928170950.png)
表面重建网络：stacked hourglass
纹理推断网络：由残差块组成的architecture of CycleGAN
隐函数网络：MLP

<!-- more -->

# Abstract

我们引入了像素对齐隐函数（PIFu），这是一种隐式表示，它将2D图像的像素与其对应的3D对象的全局上下文局部对齐。使用PIFu，我们提出了一种端到端的深度学习方法，用于数字化高度详细的穿着人类，该方法可以从单个图像和可选的多个输入图像中推断3D表面和纹理。高度复杂的形状，如发型、服装，以及它们的变化和变形，可以以统一的方式数字化。与用于3D深度学习的现有表示相比，PIFu产生了高分辨率的表面，包括基本上看不见的区域，如人的背部。特别地，与体素表示不同，它具有存储效率，可以处理任意拓扑，并且所得表面与输入图像在空间上对齐。此外，虽然以前的技术被设计为处理单个图像或多个视图，但PIFu自然地扩展到任意数量的视图。我们展示了DeepFashion数据集对真实世界图像的高分辨率和稳健重建，该数据集包含各种具有挑战性的服装类型。我们的方法在公共基准上实现了最先进的性能，并优于之前从单个图像进行人体数字化的工作。

# Method

PIFu: Pixel-Aligned Implicit Function
- $f(F(x),z(X))=s:s\in\mathbb{R},$
    - $x=\pi(X)$，2D点x是3D点X的投影
    - z(X)是相机坐标空间中的深度值
    - F(x)＝g(I(x))是x处的图像特征，双线性采样获得

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230928170950.png)

对于GT数据集，使用0.5水平集表示表面
$f_v^*(X)=\begin{cases}1,&\text{if}X\text{is inside mesh surface}\\0,&\text{otherwise}\end{cases}.$

通过Spatial Sampling.在空间中采样n个点X

**Surface Reconstruction**
$\mathcal{L}_{V}=\frac{1}{n}\sum_{i=1}^{n}|f_{v}(F_{V}(x_{i}),z(X_{i}))-f_{v}^{*}(X_{i})|^{2},$
- X为3D点，F是X对应像素x处来自编码器的图片特征

**Texture Inference**
一般$\mathcal{L}_{C}=\frac{1}{n}\sum_{i=1}^{n}|f_{c}(F_{C}(x_{i}),z(X_{i}))-C(X_{i})|,$
- $C(X_{i})$是表面点X的地面真实RGB值

使用上述损失函数天真地训练fc严重存在过拟合的问题
本文使用$\mathcal{L}_{C}=\frac{1}{n}\sum_{i=1}^{n}\big|f_{c}(F_{C}(x_{i}',F_{V}),X_{i,z}')-C(X_{i})\big|,$
- 添加几何特征输入
- 引入偏移量：$\epsilon\sim\mathcal{N}(0,d)$
    - $X_{i}^{\prime}=X_{i}+\epsilon\cdot N_{i}.$
    - d = 1.0 cm

**MVS**
将隐式函数f分解为特征嵌入函数f1和多视图推理函数f2
![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230928173658.png)

# Experiments

Datasets：RenderPeople、BUFF、DeepFashion

# Discussion

我们引入了一种新的像素对齐隐式函数，该函数将输入图像的像素级信息与3D物体的形状在空间上对齐，用于基于深度学习的3D形状和纹理推理，从单个输入图像中推断穿衣服的人。
我们的实验表明，可以推断出高度可信的几何形状，包括大部分看不见的区域，如人的背部，同时保留图像中存在的高频细节。与基于体素的表示不同，我们的方法可以产生高分辨率的输出，因为我们不受体积表示的高内存要求的限制。此外，我们还演示了如何将这种方法自然地扩展到在给定部分观察的情况下推断一个人的整个纹理。与现有的基于图像空间中的正面视图合成背面区域的方法不同，我们的方法可以直接在表面上预测未见区域、凹区域和侧面区域的颜色。
特别是，我们的方法是第一个可以为任意拓扑形状绘制纹理的方法。由于我们能够从单个RGB相机生成穿衣服的人的纹理3D表面，因此我们正在朝着无需模板模型即可从视频中单目重建动态场景的方向迈进一步。我们处理任意附加视图的能力也使我们的方法特别适合使用稀疏视图的实用和有效的3D建模设置，传统的多视图立体或运动结构将fail。

**Future Work**.
虽然我们的纹理预测是合理的，并且不受推断的3D表面的拓扑或参数化的限制，但我们相信可以推断出更高分辨率的外观，可能使用生成对抗网络或增加输入图像分辨率。在这项工作中，重建在像素坐标空间中进行，对准被试的尺度作为预处理。与其他单视图方法一样，推断尺度因子仍然是一个开放的问题，未来的工作可以解决这个问题。最后，在我们所有的例子中，没有一个被分割的主题被任何其他物体或场景元素遮挡。在现实世界中，遮挡经常发生，也许只有身体的一部分在相机中被框住。能够在部分可见的环境中对完整的物体进行数字化和预测，对于在不受约束的环境中分析人类非常有价值。