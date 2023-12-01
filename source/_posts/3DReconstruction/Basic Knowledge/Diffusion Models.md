---
title: What is Diffusion Models ?
date: 2023-10-22 20:29:02
tags:
  - Diffusion
categories: 3DReconstruction/Basic Knowledge
---
Diffusion Models 原理

<!-- more -->

[【生成式AI】Diffusion Model 原理剖析 (2/4) (optional) - YouTube](https://www.youtube.com/watch?v=73qwu77ZsTM)

生成模型共同目标：
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022203426.png)
如何优化：
**Maximum Likelihood == Minimize KL Divergence**
最大化 $P_{\theta}(x)$ 分布中从 $P_{data}(x)$ 采样出来的 $x_{i},..., x_{m}$ 的概率，相当于最小化 $P_{\theta}(x)$ 与 $P_{data}(x)$ 之间的差异 

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022203514.png)

# VAE

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022204157.png)

$P_\theta(x)=\int\limits_zP(z)P_\theta(x|z)dz$
$\begin{aligned}&P_\theta(x|\mathrm{z})\propto\exp(-\|G(\mathrm{z})-x\|_2)\end{aligned}$

Maximize ： Lower bound of $logP(x)$ 
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022204951.png)


# Diffusion Model

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022210535.png)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022210617.png)

## DDPM(Denoising Diffusion Probabilistic Models)

$\text{Maximize E}_{q(x_1:x_T|x_0)}[log\left(\frac{P(x_0;x_T)}{q(x_1:x_T|x_0)}\right)]$

$q(x_t|x_0)$ 可以只做一次 sample(给定一系列 $\beta$)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023100329.png)

DDPM 的 Lower bound of $logP(x)$ 
复杂公式推导得到：
$logP(x) \geq \operatorname{E}_{q(x_1|x_0)}[logP(x_0|x_1)]-KL\big(q(x_T|x_0)||P(x_T)\big)-\sum_{t=2}^{T}\mathrm{E}_{q(x_{t}|x_{0})}\bigl[KL\bigl(q(x_{t-1}|x_{t},x_{0})||P(x_{t-1}|x_{t})\bigr)\bigr]$

- $q(x_{t-1}|x_{t},x_{0}) =\frac{q(x_{t}|x_{t-1})q(x_{t-1}|x_{0})}{q(x_{t}|x_{0})}$ 为一个 Gaussian distribution
  -  $mean = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}x_{0}+\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})x_{t}}{1-\bar{\alpha}_{t}}$ ，$variance = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta$

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023102747.png)

## DDPM Code

[mikonvergence/DiffusionFastForward: DiffusionFastForward: a free course and experimental framework for diffusion-based generative models (github.com)](https://github.com/mikonvergence/DiffusionFastForward)

### Schedule

* `betas`: $\beta_t$ , `betas=torch.linspace(1e-4,2e-2,num_timesteps)`
* `alphas`: $\alpha_t=1-\beta_t$ 
* `alphas_sqrt`:  $\sqrt{\alpha_t}$
* `alphas_prod`: $\bar{\alpha}_t=\prod_{i=0}^{t}\alpha_i$
* `alphas_prod_sqrt`: $\sqrt{\bar{\alpha}_t}$

### Forward Process

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

### Reverse Process
至少三种逆向过程的求法，从 $x_{t}$ 到 $x_{0}$
There are **at least 3 ways of parameterizing the mean** of the reverse step distribution $p_\theta(x_{t-1}|x_t)$:
* Directly (a neural network will estimate $\mu_\theta$)直接用网络预测 $\mu_\theta$
* Via $x_0$ (a neural network will estimate $x_0$)用网络预测 $x_0$
$$\tilde{\mu}_\theta = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t\tag{4}$$
* Via noise $\epsilon$ subtraction from $x_0$ (a neural network will estimate $\epsilon$)用网络预测噪声 $\epsilon$
$$x_0=\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon)\tag{5}$$

### Test to

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


# Stable Diffusion

目前常见的 UI 有 WebUI 和 ComfyUI

## 模型

模型格式：
- 主模型 checkpoints：*ckpt, safetensors*
- 微调模型
  - LoRA 和 LyCORIS 控制画风和角色：*safetensors*
  - 文本编码器模型：*pt,safetensors*
    - Embedding 输入文本 prompt 进行编码 *pt*
  - Hypernetworks 低配版的 lora *pt*
  - ControlNet
  - VAE 图片与潜在空间 *pt*


## 采样器

[Stable diffusion采样器全解析，30种采样算法教程](https://www.bilibili.com/video/BV1FN411i7sB/)
DPM++2M Karras，收敛+速度快+质量 OK

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022195103.png)

