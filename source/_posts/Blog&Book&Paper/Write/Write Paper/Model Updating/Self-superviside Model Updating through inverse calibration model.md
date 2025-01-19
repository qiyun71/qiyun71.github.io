---
title: Stoachastic Model Updating with the Ellipse Similarity Metric
date: 2024-05-20 19:13:51
tags: 
categories:
---

| Title     | 带记忆的快速model updating by self-supervised learning network |
| --------- | -------------------------------------------------------- |
| Author    |                                                          |
| Conf/Jour |                                                          |
| Year      |                                                          |
| Project   |                                                          |
| Paper     |                                                          |

<!-- more -->

# Paper

最终目标:
根据一组time response data辨识出 aleatory(偶然，随机) 和 epistemic(认知，区间) 参数
- 思路1：随机的辨识网络，直接网络根据一组响应数据辨识一组参数
- 思路2：确定的辨识网络，网络根据单个响应数据辨识单个参数，对一组中的每个数据进行单独辨识

# Time series Surrogate Model

难点：
- 参数-->时间序列数据 的代理模型构建  [Time Series Generation | Papers With Code](https://paperswithcode.com/task/time-series-generation/latest)
  - [Diffusion TS](https://openreview.net/pdf?id=4h1apFjO99) 这种方法生成的数据并不固定，无法用作确定的代理模型
- 序列数据的评价指标

**直接使用L1Loss + MLP**，可以发现在数据包含高频成分时预测的不是很好

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240925170521.png)

**使用L1loss+FFTloss** 依然会出现这种问题，此外生成out出来的时域信号 $u(t)$ 相较于原来的label，会很粗糙(有噪声)，做 FFT后可以看出来其频域信号也会出现很多噪声

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240926103010.png)
![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240926102832.png)

**使用傅里叶思路**，预测每个谐波的振幅(amplitude)、相位(phase)和频率(frequency)，然后通过多个cos联合起来：$u(t)=\sum_{i=1}^{m}A_{i}\cos(2\pi f_{i}+\phi_{i})$ ，*结果看起来还可以*

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240926211853.png)
![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240926211901.png)


# Self-supervised model

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240927210534.png)

部分结果预测的不是很好(using FE or surrogate model)，**使用L1损失的结果**

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240927210523.png)

**改用L2损失**：

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240928195106.png)




# Ellipse loss function
## 置信度椭圆

> [主成分分析 (PCA) 的主轴和置信椭圆可视化 - 知乎](https://zhuanlan.zhihu.com/p/352715707)

对XY两方向分布进行主成分分析：
- 求XY两组数据的协方差矩阵C
- 对协方差矩阵特征值分解，求特征值和特征向量$C=Q\Lambda Q^{T}$
- 特征值$\lambda_1、\lambda_2$即为椭圆的长短两半轴长度($\lambda_1>\lambda_2$)
- 特征向量$\lambda_1$对应的特征值即为长轴的方向，可以求得长轴与x方向的夹角$\arctan \frac{y_{1}}{x_{1}}$

$\left(\frac x{\lambda_1}\right)^2+\left(\frac y{\lambda_2}\right)^2=\kappa$

$loss_{ellipse} = \lambda_{distance} loss_{distance} + \lambda_{\alpha} loss_{\alpha} + \lambda_{ab} loss_{ab}$

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240520193055.png)

- 中心距：$loss_{distance}=\sqrt{(x_1^2-x_2^2)+(y_1^2-y_2^2)}$
- 主轴角度：$loss_\alpha=min(|\alpha_1-\alpha_2|,180-|\alpha_1-\alpha_2|)$ ($-90 \degree <\alpha<90\degree$)
- 轴长：$loss_{ab}=|a_1-a_2|+|b_1-b_2|$

## Question

基于置信度椭圆相似度提出的loss_ellipse，对VGG进行训练，神经网络预测的二维分布点大多数点会聚集在中心，然后周围用少量点与中心点组合后，形成的置信度椭圆与实际分布生成的置信椭圆也可以逐渐收敛到近似。

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240520191829.png)


## Solve


### 数据处理问题？

**归一化处理不对**
- 之前的归一化处理方式：将1000组的每组60x5的数据单独进行归一化处理，1000组60x5的图片可能会有重合的部分，导致ab一对多frequency的情况发生
- ***修改后的归一化方式***：将1000组所有的60x5数据统一进行归一化处理，1000组图片的值大的还是大，小的还是小，不会出现ab一对多frequency的情况发生(**只要nastran生成的数据没有问题**)


### ~~Chamfer Distance Loss~~

> [pytorch3d.loss — PyTorch3D documentation](https://pytorch3d.readthedocs.io/en/latest/modules/loss.html) pytorch3d.loss.chamfer_distance
> [【常用损失函数】 L1Loss｜CrossEntropyLoss｜NLLLoss｜MSELoss｜DiceLoss｜Focal Loss｜Chamfer Distance｜smooth L1 loss - 知乎](https://zhuanlan.zhihu.com/p/401010037)

倒角距离CD：$d_{\mathrm{CD}}(S_1,S_2)=\frac{1}{S_1}\sum_{x\in S_1}\min_{y\in S_2}\|x-y\|_2^2+\frac{1}{S_2}\sum_{y\in S_2}\min_{x\in S_1}\|y-x\|_2^2$

$S_{1}$中每一点与$S_{2}$中最近一点的距离之和 + $S_{2}$中每一点与$S_{1}$中最近一点的距离之和


### Sub Interval Similarity

区间相似度 --> 子区间相似度(缺点：时间慢5 x other_time)

区间重合度问题，两个区间$A = [\underline{a},\overline{a}],B = [\underline{b},\overline{b}]$：

$\left.\mathrm{RPO}(A,B)=\begin{cases}\frac{(\overline{a}-\underline{b})}{\max\{L(A),L(B)\}},&\text{Case 1,2}\\\\\frac{(\overline{a}-\underline{a})}{\max\{L(A),L(B)\}},&\text{Case 3}\\\\\frac{(\overline{b}-\underline{b})}{\max\{L(A),L(B)\}},&\text{Case 4}\\\\\frac{(\overline{b}-\underline{a})}{\max\{L(A),L(B)\}},&\text{Case 5,6}&\end{cases}\right.$

Relative Position Operator (RPO) $\mathrm R\mathrm P\mathrm O(\mathrm A,\mathrm B)\in(-\infty,1]$ **1最好**
Interval Similarity Function (ISF) $\mathrm{ISF}(A,B)=\frac{1}{1+\exp\{-\mathrm{RPO}(A,B)\}}$ $ISF(A,B) \in (0,0.7311]$ **0.7311最好**
$Loss_{ISF} = \frac{1}{1+\exp\{-1\}} -\frac{1}{1+\exp\{-\mathrm{RPO}(A,B)\}}$ $Loss_{ISF} \in [0,0.7311)$ **0最好**

**更简单的** $Loss_{ISF} = 1- \mathrm{RPO}(A,B) = 1-\frac{min\{\overline{a},\overline{b}\}-max\{\underline{a},\underline{b}\}}{max\{len(A),len(B)\}}$

根据 $SIS(A,B)|n_{sub}=\frac{1}{n_{sub}}\sum_{j=1}^{n_{sub}}\bigl\{1-ISF\bigl(A^{(j)},B^{(j)}\bigr)\bigr\}$，$Loss_{SIS} = \frac{1}{n_{sub}} \sum_{j}^{n_{sub}}Loss_{ISF}^{j}$


![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240522161610.png)

#### 问题：(只观察了很短的epoch，试试多训练几个epoch)
以case1为例，当网络prediction的A与target的B出现$len(A) >> |\overline{a} - \underline{b}| >len(B)$时，$RPO \approx -0$，这时$ISF \approx 0.5$，$Loss_{ISF} \approx 0.2311$ 陷入局部最优，这时候考虑$Loss_{SIS}$，除了中间地方，其他子区间的$Loss_{ISF} \approx 0.2311$，网络只能通过优化中间部位的子区间来降低loss

**为什么网络总是会预测出一个异常点？**

|             | sub int (less epoch)                                                                                              | sub int (more epoch)                                                                                              |
| ----------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Bad Result  | ![image.png\|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240523113736.png) | ![image.png\|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240523113736.png) |
| Description |                                                                                                                   |                                                                                                                   |




