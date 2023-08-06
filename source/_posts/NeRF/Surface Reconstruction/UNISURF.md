---
title: UNISURF
date: 2023-08-06 14:32:58
tags:
    - Surface Reconstruction
    - UNISURF
categories: NeRF/Surface Reconstruction
---

[UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction (moechsle.github.io)](https://moechsle.github.io/unisurf/)

[UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4546355486135050241&noteId=1791178045241021696)
![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230806143334.png)


<!-- more -->


# AIR

神经隐式三维表示已经成为从多视图图像重建表面和合成新视图的强大范例。**不幸的是，现有的方法，如DVR或IDR需要精确的过像素对象掩码Mask作为监督**。
与此同时，神经辐射场已经彻底改变了新的视图合成。**然而，NeRF的估计体积密度不允许精确的表面重建**。
我们的关键见解是隐式表面模型和亮度场可以以统一的方式制定，使表面和体渲染使用相同的模型。这种统一的视角使新颖，更有效的采样程序和重建精确表面的能力无需输入掩模。我们在DTU、BlendedMVS和合成室内数据集上比较了我们的方法。我们的实验表明，我们在重建质量方面优于NeRF，同时**在不需要掩模的情况下与IDR表现相当**。


## Introduction

从一组图像中捕捉3D场景的几何形状和外观是计算机视觉的基础问题之一。为了实现这一目标，基于坐标的神经模型在过去几年中已经成为三维几何和外观重建的强大工具。
最近的许多方法使用连续隐式函数参数化神经网络作为几何图形的三维表示[3,8,12,31,32,37,41,43,47,57]或外观[34,38,39,40,47,52,61]。这些神经网络三维表示在多视图图像的几何重建和新视图合成方面显示出令人印象深刻的性能。神经隐式多视图重建除了选择三维表示形式(如占用场、无符号距离场或有符号距离场)外，渲染技术是实现多视图重建的关键。虽然其中一些作品将隐式表面表示为水平集，从而渲染表面的外观[38,52,61]，但其他作品通过沿着观察光线绘制样本来整合密度[22,34,49]。
在现有的工作中，表面渲染技术在三维重建中表现出了令人印象深刻的性能[38,61]。**然而，它们需要逐像素对象掩码作为输入和适当的网络初始化，因为表面渲染技术只在表面与射线相交的局部提供梯度信息**。直观地说，optimizing wrt. 局部梯度可以看作是应用于初始神经表面的迭代变形过程，初始神经表面通常被初始化为一个球体。为了收敛到一个有效的表面，需要额外的约束，如掩码监督，如图2所示。
==现有工作2021由于依赖mask，因此只能用于对象级重建，而无法重建大场景==

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230806160553.png)

