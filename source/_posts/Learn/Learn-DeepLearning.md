---
title: 深度学习基础
date: 2023-07-02 18:53:58
tags:
    - DeepLearning
categories: Learn
---

学习深度学习过程中的一些基础知识

| DL     | 修仙.炼丹 | example |
| ------ | --------- | ------- |
| 框架   | 丹炉      | PyTorch |
| 网络   | 丹方.灵阵 | CNN     |
| 数据集 | 灵材      | MNIST   |
| GPU    | 真火      | NVIDIA  |
| 模型   | 成丹      | .ckpt        |

>[深度学习·炼丹入门 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/23781756)


<!-- more -->

学习顺序MLP-RNN-seq2seq/编码器解码器架构-注意力机制-自注意力-transformer。

# Model

## CNN

```text
卷积神将网络的计算公式为：
N=(W-F+2P)/S+1
其中
N：输出大小
W：输入大小
F：卷积核大小
P：填充值的大小
S：步长大小
```

### UCNN

### VGG16

[VGG16学习笔记 | 韩鼎の个人网站 (deanhan.com)](https://deanhan.com/2018/07/26/vgg16/)

### ResNet

[ResNet中的BasicBlock与bottleneck-CSDN博客](https://blog.csdn.net/sazass/article/details/116864275)

## RNN

相比一般的神经网络来说，他能够处理序列变化的数据

### LSTM

[人人都能看懂的LSTM - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/32085405)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240112195022.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240112194924.png)

## U-Net

> [U-Net (labml.ai)](https://nn.labml.ai/unet/index.html)


图像分割

# Loss

损失函数在训练过程中，突然变得很大或者nan
添加 torch.cuda.amp.GradScaler() 解决 loss为nan或inf的问题

# 神经网络MLP

## 前向传播

根据每层的输入、权重weight和偏置bias，求出该层的输出，然后经过激活函数。按此一层一层传递，最终获得输出层的输出。

## 反向传播

>[神经网络之反向传播算法（BP）公式推导（超详细） - jsfantasy - 博客园 (cnblogs.com)](https://www.cnblogs.com/jsfantasy/p/12177275.html)

假如激活函数为sigmoid函数：$\sigma(x) = \frac{1}{1+e^{-x}}$
sigmoid的导数为：$\frac{d}{dx}\sigma(x) = \frac{d}{dx} \left(\frac{1}{1+e^{-x}} \right)= \sigma(1-\sigma)$

因此当损失函数对权重求导，其结果与sigmoid的输出相关

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230702194201.png)

- o代表输出，上标表示当前的层数，下标表示当前层数的第几号输出
- z代表求和结果，即sigmoid的输入
- 权重$w^{J}_{ij}$的上标表示权值所属的层数，下标表示从I层的第i号节点到J层的第j号节点

输出对J层的权重$w_{ij}$求导： 

$$\begin{align*}
\frac{\partial L}{\partial w_{ij}} &=\frac{\partial}{\partial w_{ij}}\frac{1}{2}\sum_{k}(o_{k}-t_{k})^{2} \\
&= \sum_k(o_k-t_k)\frac{\partial o_k}{\partial w_{ij}}\\
&= \sum_k(o_k-t_k)\frac{\partial \sigma(z_k)}{\partial w_{ij}}\\
&= \sum_k(o_k-t_k)o_k(1-o_k)\frac{\partial z_k}{\partial w_{ij}}\\
&= \sum_k(o_k-t_k)o_k(1-o_k)w_{jk}\cdot\frac{\partial o_j}{\partial w_{ij}}
\end{align*}
$$

$\frac{\partial z_k}{\partial w_{ij}} = \frac{\partial z_k}{o_j}\cdot \frac{\partial o_j}{\partial w_{ij}} = w_{jk} \cdot \frac{\partial o_j}{\partial w_{ij}}$, because $z_k = o_j \cdot w_{jk} + b_k$

and $\frac{\partial z_j}{\partial w_{ij}} = o_i \left(z_j = o_i\cdot w_{ij} + b_j\right)$

$$\begin{align*}
\frac{\partial L}{\partial w_{ij}} 
&= \sum_k(o_k-t_k)o_k(1-o_k)w_{jk}\cdot\frac{\partial o_j}{\partial w_{ij}}\\
&= \frac{\partial o_j}{\partial w_{ij}}\cdot\sum_k(o_k-t_k)o_k(1-o_k)w_{jk}\\
&= o_j(1-o_j)\frac{\partial z_j}{\partial w_{ij}} \cdot\sum_k(o_k-t_k)o_k(1-o_k)w_{jk}\\
&= o_j(1-o_j)o_i \cdot\sum_k(o_k-t_k)o_k(1-o_k)w_{jk}\\
&= o_j(1-o_j)o_i \cdot\sum_k\delta _k^K\cdot w_{jk}\\
&= \delta_j^J\cdot o_i^I
\end{align*}
$$
其中 $\delta_j^J = o_j(1-o_j) \cdot \sum_k \delta _k^K\cdot w_{jk}$


推广：
- 输出层：$\frac{\partial L}{\partial w_{jk}} = \delta _k^K\cdot o_j$ ,其中$\delta _k^K = (o_k-t_k)o_k(1-o_k)$
- 倒二层：$\frac{\partial L}{\partial w_{ij}} = \delta _j^J\cdot o_i$ ,其中$\delta_j^J = o_j(1-o_j) \cdot \sum_k \delta _k^K\cdot w_{jk}$
- 倒三层：$\frac{\partial L}{\partial w_{ni}} = \delta _i^I\cdot o_n$ ,其中$\delta_i^I = o_i(1-o_i)\cdot \sum_j\delta_j^J\cdot w_{ij}$
    - $o_n$ 为倒三层输入，即倒四层的输出

根据每一层的输入或输出，以及真实值，即可计算loss对每个权重参数的导数

## 优化算法

>[11.1. 优化和深度学习 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh.d2l.ai/chapter_optimization/optimization-intro.html)

不同的算法有不同的参数更新方式

### 优化的目标

训练数据集的最低经验风险可能与最低风险（泛化误差）不同
- 经验风险是训练数据集的平均损失
- 风险则是整个数据群的预期损失

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230702201639.png)

### 优化的挑战


<table>
  <tr>
    <td style="text-align:center;">
      <img src="https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230702201744.png" alt="Image 1" style="width:500px;">
      <p>局部最优</p>
    </td>
    <td style="text-align:center;">
      <img src="https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230702201754.png" alt="Image 2" style="width:500px;">
      <p>鞍点</p>
    </td>
        <td style="text-align:center;">
      <img src="https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230702202522.png" alt="Image 2" style="width:500px;">
      <p>梯度消失</p>
    </td>
  </tr>
</table>

(鞍点 in 3D)saddle point be like: 
![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230702201813.png)



# 位置编码

Position Embedding 与 Position encoding的区别

> [两个PE的不同](https://www.zhihu.com/question/402387099/answer/1366825959  )


position embedding：随网络一起训练出来的位置向量，与前面说的一致，可以理解成动态的，即每次训练结果可能不一样。

position encoding：根据一定的编码规则计算出来位置表示，比如

$$\gamma(p)=\left(\sin \left(2^{0} \pi p\right), \cos \left(2^{0} \pi p\right), \cdots, \sin \left(2^{L-1} \pi p\right), \cos \left(2^{L-1} \pi p\right)\right)$$


# 迁移学习

迁移学习通常会关注有一个源域 $D_{s}$ 和一个目标域$D_{t}$ 的情况，将源域中网络学习到的知识迁移到目标域的学习中

[Transfer learning 【迁移学习综述_汇总】 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/443079853)
