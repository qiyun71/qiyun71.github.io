---
title: Math
date: 2023-10-12 21:09:01
tags:
  - Math
  - 3DReconstruction
  - SurfaceReconstruction
categories: HumanBodyReconstruction
---

3D Reconstruction 相关数学方法

<!-- more -->

# 权重计算

## 2K2K

来源：[2K2K Code](https://github.com/SangHunHan92/2K2K/blob/main/models/deep_human_models.py)
目的：part 法向量图 --> 原图大小对应法向量图

根据 part 法向量图逆仿射变换回原图空间 $\mathbf{n}_{i}=\mathbf{M}_{i}^{-1}\mathbf{\bar{n}}_{i}$
要将 part 法向量图融合为原图空间法向量图，每个法向量图有不同的权重$\mathbf{N}^h\quad=\sum\limits_{i=1}^K\left(\mathbf{W}_i\odot\mathbf{n}_i\right)$

权重的**计算方法**：$\mathbf{W}_i(x,y)=\frac{G(x,y)*\phi_i(x,y)}{\sum_iG(x,y)*\phi_i(x,y)}$

- 同时与 part 法向量图逆仿射变换的还有一个 Occupancy Grid Map O，表示在原图空间中每个 part 的占用值 0 或者 1，i.e. $\left.\phi_i(x,y)=\left\{\begin{array}{cc}1&\text{if}&\sum\mathbf{n}_i(x,y)\neq\mathbf{0}^\top\\0&\text{otherwise}\end{array}\right.\right.$
- 对 O 做高斯模糊 GaussianBlur，**使得 O map 的值到边缘逐渐减小**
- 如下图，face part 脖子上方中心处 O 值做完高斯模糊后依然近似 1(假设 1)，而 body part 上部分脖子中心处做完高斯模糊后 O 值<1(假设 1/2)，这会导致对于脖子这部分多 part 融合时，face part normal 的权重相对于 body part normal 的权重会更大一点(2/3 > 1/3)

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230921163941.png)
