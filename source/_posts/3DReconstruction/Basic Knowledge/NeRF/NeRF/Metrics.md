---
title: Metrics
date: 2023-10-10 10:23:09
tags:
  - Metrics
categories: 3DReconstruction/Basic Knowledge/NeRF/NeRF
---

NeRF、Depth Estimation、Object Detection
- 评价指标
- Loss损失

<!-- more -->

评价指标代码
- [NeRF与三维重建专栏（一）领域背景、难点与数据集介绍 - 掘金 (juejin.cn)](https://juejin.cn/post/7232499180659458109)

论文：(很少)
- Towards a Robust Framework for NeRF Evaluation

# Metrics

L1*loss : $loss(x,y)=\frac{1}{n}\sum*{i=1}^{n}|y*i-f(x_i)|$
L2_loss: $loss(x,y)=\frac{1}{n}\sum*{i=1}^{n}(y_i-f(x_i))^2$

在标准设置中通过 NeRF 进行的新颖视图合成使用了视觉质量评估指标作为基准。这些指标试图评估单个图像的质量，要么有(完全参考)，要么没有(无参考)地面真相图像。峰值信噪比(PSNR)，结构相似指数度量(SSIM)[32]，学习感知图像补丁相似性(LPIPS)[33]是目前为止在 NeRF 文献中最常用的。

[有真实参照的图像质量的客观评估指标:SSIM、PSNR和LPIPS - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/309892873)

## PSNR↑

峰值信噪比 Peak Signal to Noise Ratio
PSNR 是一个无参考的质量评估指标
$PSNR(I)=10\cdot\log_{10}(\dfrac{MAX(I)^2}{MSE(I)})$
$MSE=\frac1{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1}[I(i,j)-K(i,j)]^2$
$MAX(I)^{2}$（动态范围可能的最大像素值，b 位：$2^{b}-1$），eg: 8 位图像则$MAX(I)^{2} = 255$

```python
# Neus
psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

# instant-nsr-pl
psnr = -10. * torch.log10(torch.mean((pred_rgb.to(gt_rgb)-gt_rgb)**2))
```

## SSIM↑

[VainF/pytorch-msssim: Fast and differentiable MS-SSIM and SSIM for pytorch. (github.com)](https://github.com/VainF/pytorch-msssim)

结构相似性 Structural Similarity Index Measure
SSIM 是一个完整的参考质量评估指标。
$SSIM(x,y)=\dfrac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}$
衡量了两张图片之间相似度：($C_1,C_2$为常数防止除以 0)

$S(x,y)=l(x,y)^{\alpha}\cdot c(x,y)^{\beta}\cdot s(x,y)^{\gamma}$

$C_1=(K_1L)^2,C_2=(K_2L)^2,C_3=C_2/2$
$K_{1}= 0.01 , K_{2} = 0.03 , L = 2^{b}-1$

- 亮度，图像 x 与图像 y 亮度 $l(x,y) =\frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2+C_1}$
  - $\mu_{x} =\frac1N\sum_{i=1}^Nx_i$像素均值
    - $x_i$像素值，N 总像素数
  - 当 x 与 y 相同时，$l(x,y) = 1$
- 对比度，$c(x,y)=\frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2+\sigma_y^2+C_2}$
  - 图像标准差$\sigma_x=(\frac1{N-1}\sum_{i=1}^N(x_i-\mu_x)^2)^{\frac12}$
- 结构对比，$s(x,y)=\frac{\sigma_{xy}+C_3}{\sigma_x\sigma_y+C_3}$ - 图像的协方差$\sigma_{xy}=\frac1{N-1}\sum_{i=1}^N(x_i-\mu_x)(y_i-\mu_y)$
  实际使用中(圆对称高斯加权公式)，使用一个高斯核对局部像素求 SSIM，最后对所有的局部 SSIM 求平均得到 MSSIM

使用高斯核，均值、标准差和协方差变为：
$\mu_{x}=\sum_{i}w_{i}x_{i}$
$\sigma_{x}=(\sum_{i}w_{i}(x_{i}-\mu_{x})^{2})^{1/2}$
$\sigma_{xy}=\sum_{i}w_{i}(x_{i}-\mu_{x})(y_{i}-\mu_{y})$

## LPIPS↓

学习感知图像块相似度 Learned Perceptual Image Patch Similarity
**LPIPS 比传统方法（比如 L2/PSNR, SSIM, FSIM）更符合人类的感知情况**。**LPIPS 的值越低表示两张图像越相似，反之，则差异越大。**
![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230801170138.png)

LPIPS 是一个完整的参考质量评估指标，它使用了学习的卷积特征。分数是由多层特征映射的加权像素级 MSE 给出的。
$LPIPS(x,y)=\sum\limits_{l}^{L}\dfrac{1}{H_lW_l}\sum\limits_{h,w}^{H_l,W_l}||w_l\odot(x^l_{hw}-y^l_{hw})||^2_2$

## CD↓

[jzhangbs/DTUeval-python: A fast python implementation of DTU MVS 2014 evaluation (github.com)](https://github.com/jzhangbs/DTUeval-python)

Chamfer Distance 倒角距离
点云或 mesh 重建模型评估指标，它度量两个点集之间的距离，其中一个点集是参考点集，另一个点集是待评估点集

$d_{\mathrm{CD}}(S_1,S_2)=\frac{1}{S_1}\sum_{x\in S_1}\min_{y\in S_2}\lVert x-y\rVert_2^2+\frac{1}{S_2}\sum_{y\in S_2}\min_{x\in S_1}\lVert y-x\rVert_2^2$

S1 和 S2 分别表示两组 3D 点云，第一项代表 S1 中任意一点 x 到 S2 的最小距离之和，第二项则表示 S2 中任意一点 y 到 S1 的最小距离之和。
如果该距离较大，则说明两组点云区别较大；如果距离较小，则说明重建效果较好。

$\begin{aligned}\mathcal{L}_{CD}&=\sum_{y'\in Y'}min_{y\in Y}||y'-y||_2^2+\sum_{y\in Y}min_{y'\in Y'}||y-y'||_2^2,\end{aligned}$

## P2S↓

average point-to-surface(P2S) distance平均点到面距离

**P2S距离：** CAPE数据集scan包含大的空洞，为了排除孔洞影响，我们记录scan点到最近重构表面点之间距离，为Chamfer距离的单向版本；measure the average point-to-surface Euclidean distance (P2S) in cm **from the vertices on the reconstructed surface to the ground truth**

## Normal↓

average surface normal error平均表面法向损失

**Normal difference:** 表示使用重构的及真值surface分别进行渲染normal图片，计算两者之间L2距离，用于捕获高频几何细节误差。
For both reconstructed and ground truth surfaces, we **render their normal maps** in the image space from the input viewpoint respectively. We then **calculate the L2 error** between these two normal maps.

## IoU↑

Intersection over Union(IoU)交并比
在目标检测中用到的指标$IOU = \frac{A \cap B}{A \cup B}$ 
一般来说，这个比值 ＞ 0.5 就可以认为是一个不错的结果了。
- A: GT bounding box
- B: Predicted bounding box

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231010101549.png)

## EMD↓

Earth Mover's distance 推土距离,度量两个分布之间的距离
[EMD(earth mover's distances)距离 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/145739750) 

$\mathcal{L}_{EMD}=min_{\phi:Y\rightarrow Y^{\prime}}\sum_{x\in Y}||x-\phi(x)||_{2}$ , φ indicates a parameter of bijection.


# Loss

## RGB Loss

L2 损失：`F.mse_loss(pred_rgb, gt_rgb)` $L=\sum_{i=1}^n(y_i-f(x_i))^2$
L1 损失：`F.l1_loss(pred_rgb, gt_rgb)`更稳定？ $L=\sum_{i=1}^n|y_i-f(x_i)|$

## Eikonal Loss

$\mathcal{L}_{r e g}=\frac{1}{n m}\sum_{k,i}(\|\nabla f(\hat{\mathbf{p}}_{k,i})\|_{2}-1)^{2}.$

## Mask Loss

$\mathcal{L}_{mask}=\mathrm{BCE}(M_k,\hat{O}_k)$

- $\hat{O}_k=\sum_{i=1}^n T_{k,i}\alpha_{k,i}$
- $M_{k} ∈ {0, 1}$

BCE 二值交叉熵损失：让输出$\hat{O}_k$去逼近 label $M_{k}$

> 一种新的 BCE loss[ECCV'22 ｜ Spatial-BCE - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/593711934)

## Opacity Loss

`loss_opaque = -(opacity * torch.log(opacity) + (1 - opacity) * torch.log(1 - opacity)).mean()`
$opaque = BCE(opaque,opaque) = -[opaque * ln(opaque) + (1-opaque) *ln(1-opaque)]$

使得 opacity 更加接近 0 或者 1

## Sparsity Loss

`loss_sparsity = torch.exp(-self.conf.loss.sparsity_scale * out['sdf_samples'].abs()).mean()`
$sparsity = \frac{1}{N} \sum e^{-scale * sdf}$
让 sdf 的平均值更小，前景物体更加稀疏，物体内的点往外发散

## Geo-Neus

- sdf loss
  - `sdf_loss = F.l1_loss(pts2sdf, torch.zeros_like(pts2sdf), reduction='sum') / pts2sdf.shape[0]`
  - $\mathcal{L}_{sdf} = \frac{1}{N} \sum |sdf(spoint) - 0|$


## other loss

### 加强Eikonal对SDF的优化

[sunyx523/StEik (github.com)](https://github.com/sunyx523/StEik)
[NeurIPS 2023 | 三维重建中的Neural SDF(Neural Implicit Surface) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/649921965)
一个好的SDF其实只需要其法线方向上的二阶导数为0，如果在切线方向上的二阶导数为0的话，得到的SDF轮廓会非常平滑，不利于学习到一些细节。

$L_\text{L. n.}(u)=\int_{\Omega}|\nabla u(x)^TD^2u(x)\cdot\nabla u(x)|dx.$


### S3IM Loss

[S3IM (madaoer.github.io)](https://madaoer.github.io/s3im_nerf/)

$\begin{aligned}L_{\mathrm{S3IM}}(\Theta,\mathcal{R})=&1-\mathrm{S3IM}(\hat{\mathcal{R}},\mathcal{R})=1-\frac{1}{M}\sum_{m=1}^{M}\mathrm{SSIM}(\mathcal{P}^{(m)}(\hat{\mathcal{C}}),\mathcal{P}^{(m)}(\mathcal{C})).\end{aligned}$

### Smoothness Loss

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230919194046.png)

### LCN

Learnable Chamfer Distance for Point Cloud Reconstruction
我们提出了一种简单但有效的重建损失，称为可学习倒角距离（LCD），通过动态关注由一组可学习网络控制的不同权重分布的匹配距离

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240104172937.png)
