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

# 易错

## 归一化

**输入数据**的归一化：
当输入的数据为多个变量时，如果某个变量的变化远大于其他变量的变化，则会出现大数吃小数问题，网络会按着变化大的变量(对其最为敏感)来预测。

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
讲的很好：[三分钟吃透RNN和LSTM神经网络](https://www.zhihu.com/tardis/zm/art/86006495?source_id=1003)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240112195022.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240112194924.png)

## U-Net

> [U-Net (labml.ai)](https://nn.labml.ai/unet/index.html)


图像分割

## VAE

[【生成式AI】Diffusion Model 原理剖析 (2/4) (optional) - YouTube](https://www.youtube.com/watch?v=73qwu77ZsTM)

生成模型共同目标：
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022203426.png)
如何优化：
**Maximum Likelihood == Minimize KL Divergence**
最大化 $P_{\theta}(x)$ 分布中从 $P_{data}(x)$ 采样出来的 $x_{i},..., x_{m}$ 的概率，相当于最小化 $P_{\theta}(x)$ 与 $P_{data}(x)$ 之间的差异 

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022203514.png)


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022204157.png)

$P_\theta(x)=\int\limits_zP(z)P_\theta(x|z)dz$
$\begin{aligned}&P_\theta(x|\mathrm{z})\propto\exp(-\|G(\mathrm{z})-x\|_2)\end{aligned}$

Maximize ： Lower bound of $logP(x)$ 
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022204951.png)


## Diffusion Model

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022210535.png)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022210617.png)

### DDPM(Denoising Diffusion Probabilistic Models)

$\text{Maximize E}_{q(x_1:x_T|x_0)}[log\left(\frac{P(x_0;x_T)}{q(x_1:x_T|x_0)}\right)]$

$q(x_t|x_0)$ 可以只做一次 sample(给定一系列 $\beta$)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023100329.png)

DDPM 的 Lower bound of $logP(x)$ 
复杂公式推导得到：
$logP(x) \geq \operatorname{E}_{q(x_1|x_0)}[logP(x_0|x_1)]-KL\big(q(x_T|x_0)||P(x_T)\big)-\sum_{t=2}^{T}\mathrm{E}_{q(x_{t}|x_{0})}\bigl[KL\bigl(q(x_{t-1}|x_{t},x_{0})||P(x_{t-1}|x_{t})\bigr)\bigr]$

- $q(x_{t-1}|x_{t},x_{0}) =\frac{q(x_{t}|x_{t-1})q(x_{t-1}|x_{0})}{q(x_{t}|x_{0})}$ 为一个 Gaussian distribution
  -  $mean = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}x_{0}+\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})x_{t}}{1-\bar{\alpha}_{t}}$ ，$variance = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta$

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023102747.png)

### DDPM Code

[mikonvergence/DiffusionFastForward: DiffusionFastForward: a free course and experimental framework for diffusion-based generative models (github.com)](https://github.com/mikonvergence/DiffusionFastForward)

#### Schedule

* `betas`: $\beta_t$ , `betas=torch.linspace(1e-4,2e-2,num_timesteps)`
* `alphas`: $\alpha_t=1-\beta_t$ 
* `alphas_sqrt`:  $\sqrt{\alpha_t}$
* `alphas_prod`: $\bar{\alpha}_t=\prod_{i=0}^{t}\alpha_i$
* `alphas_prod_sqrt`: $\sqrt{\bar{\alpha}_t}$

#### Forward Process

Forward step:
$$q(x_t|x_{t−1}) := \mathcal{N}(x_t; \sqrt{1 − \beta_t}x_{t−1}, \beta_tI) \tag{1}$$
Forward jump:
$$q(x_t|x_0) = \mathcal{N}(x_t;\sqrt{\bar{\alpha_t}}x_0, (1 − \bar{\alpha_t})I) \tag{2}$$

```python
def forward_step(t, condition_img, return_noise=False):
    """
        forward step: t-1 -> t
    """
    assert t >= 0

    mean=alphas_sqrt[t]*condition_img
    std=betas[t].sqrt()

    # sampling from N
    if not return_noise:
        return mean+std*torch.randn_like(img)
    else:
        noise=torch.randn_like(img)
        return mean+std*noise, noise

def forward_jump(t, condition_img, condition_idx=0, return_noise=False):
    """
        forward jump: 0 -> t
    """
    assert t >= 0

    mean=alphas_cumprod_sqrt[t]*condition_img
    std=(1-alphas_cumprod[t]).sqrt()

    # sampling from N
    if not return_noise:
        return mean+std*torch.randn_like(img)
    else:
        noise=torch.randn_like(img)
        return mean+std*noise, noise
```

#### Reverse Process
至少三种逆向过程的求法，从 $x_{t}$ 到 $x_{0}$
There are **at least 3 ways of parameterizing the mean** of the reverse step distribution $p_\theta(x_{t-1}|x_t)$:
* Directly (a neural network will estimate $\mu_\theta$)直接用网络预测 $\mu_\theta$
* Via $x_0$ (a neural network will estimate $x_0$)用网络预测 $x_0$
$$\tilde{\mu}_\theta = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t\tag{4}$$
* Via noise $\epsilon$ subtraction from $x_0$ (a neural network will estimate $\epsilon$)用网络预测噪声 $\epsilon$
$$x_0=\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon)\tag{5}$$

## Stable Diffusion

目前常见的 UI 有 WebUI 和 ComfyUI

### 模型

模型格式：
- 主模型 checkpoints：*ckpt, safetensors*
- 微调模型
  - LoRA 和 LyCORIS 控制画风和角色：*safetensors*
  - 文本编码器模型：*pt,safetensors*
    - Embedding 输入文本 prompt 进行编码 *pt*
  - Hypernetworks 低配版的 lora *pt*
  - ControlNet
  - VAE 图片与潜在空间 *pt*


### 采样器

[Stable diffusion采样器全解析，30种采样算法教程](https://www.bilibili.com/video/BV1FN411i7sB/)
DPM++2M Karras，收敛+速度快+质量 OK

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022195103.png)


## MLP

### 前向传播

根据每层的输入、权重weight和偏置bias，求出该层的输出，然后经过激活函数。按此一层一层传递，最终获得输出层的输出。

### 反向传播

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

### 优化算法

>[11.1. 优化和深度学习 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh.d2l.ai/chapter_optimization/optimization-intro.html)

不同的算法有不同的参数更新方式

#### 优化的目标

训练数据集的最低经验风险可能与最低风险（泛化误差）不同
- 经验风险是训练数据集的平均损失
- 风险则是整个数据群的预期损失

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230702201639.png)

#### 优化的挑战


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


# Loss

损失函数在训练过程中，突然变得很大或者nan
添加 torch.cuda.amp.GradScaler() 解决 loss为nan或inf的问题


# 其他概念

## KL 散度

[KL 散度（相对熵） - 小时百科 (wuli.wiki)](https://wuli.wiki/online/KLD.html)

**KL 散度**（Kullback–Leibler divergence，缩写 KLD）是一种统计学度量，**表示的是一个概率分布相对于另一个概率分布的差异程度**，在信息论中又称为**相对熵**（Relative entropy）。
$$\begin{equation}
D_{KL}(P||Q)=\sum_{x\in X}P(x)ln(\frac{P(x)}{Q(x)})=\sum_{x\in X}P(x)(ln(P(x))-ln(Q(x)))~.
\end{equation}$$

对于连续型随机变量，设概率空间 X 上有两个概率分布 P 和 Q，其概率密度分别为 p 和 q，那么，P 相对于 Q 的 KL 散度定义如下：
$$\begin{equation}
D_{KL}(P||Q)=\int_{-\infty}^{+\infty}p(x)ln(\frac{p(x)}{q(x)})dx~.
\end{equation}$$

 显然，当 P=Q 时，$D_{KL}=0$

两个一维高斯分布的 KL 散度公式：
> [KL散度(Kullback-Leibler Divergence)介绍及详细公式推导 | HsinJhao's Blogs](https://hsinjhao.github.io/2019/05/22/KL-DivergenceIntroduction/)

$$\begin{aligned}
KL(p,q)& =\int[\left.p(x)\log(p(x))-p(x)\log(q(x))\right]dx  \\
&=-\frac12\left[1+\log(2\pi\sigma_1^2)\right]-\left[-\frac12\log(2\pi\sigma_2^2)-\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}\right] \\
&=\log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac12
\end{aligned}$$

## 参数重整化

[DDPMb站视频](https://www.bilibili.com/video/BV1b541197HX/)公式推导

从高斯分布中直接采样一个值出来是不可导的，无法进行梯度传递，需要进行参数重整化：
从 $\mathcal{N}(0,1)$ 中随机采样出来 z，然后对 z 做 $\mu + z * \sigma$ 相当于从高斯分布 $\mathcal{N}(\mu,\sigma)$ 中采样


##  位置编码

Position Embedding 与 Position encoding的区别

> [两个PE的不同](https://www.zhihu.com/question/402387099/answer/1366825959  )


position embedding：随网络一起训练出来的位置向量，与前面说的一致，可以理解成动态的，即每次训练结果可能不一样。

position encoding：根据一定的编码规则计算出来位置表示，比如

$$\gamma(p)=\left(\sin \left(2^{0} \pi p\right), \cos \left(2^{0} \pi p\right), \cdots, \sin \left(2^{L-1} \pi p\right), \cos \left(2^{L-1} \pi p\right)\right)$$


# 迁移学习

迁移学习通常会关注有一个源域 $D_{s}$ 和一个目标域$D_{t}$ 的情况，将源域中网络学习到的知识迁移到目标域的学习中

[Transfer learning 【迁移学习综述_汇总】 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/443079853)

# 集成学习

[集成学习（Ensemble Learning) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/27689464)

