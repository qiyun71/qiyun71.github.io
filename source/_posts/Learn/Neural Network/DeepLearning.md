---
title: 深度学习/神经网络笔记
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

# 技巧

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

单向卷积

### BCNN

> [kumar-shridhar/PyTorch-BayesianCNN: Bayesian Convolutional Neural Network with Variational Inference based on Bayes by Backprop in PyTorch.](https://github.com/kumar-shridhar/PyTorch-BayesianCNN)
> [Bayesian neural network introduction - 知乎](https://zhuanlan.zhihu.com/p/79715409)
> [重参数 (Reparameterization)-CSDN博客](https://blog.csdn.net/weixin_42437114/article/details/125671285) [漫谈重参数：从正态分布到Gumbel Softmax - 科学空间|Scientific Spaces](https://kexue.fm/archives/6705)


初始化的权重参数用高斯先验分布表示：$p(w^{(i)})=\prod_i\mathcal{N}(w_i|0,\sigma_p^2)$，训练的过程就是根据权重先验和数据集来获得权重参数的后验分布：$p(w|\mathcal{D})=\frac{p(\mathcal{D}|w)p(w)}{p(\mathcal{D})}$

![CNNwithdist.png|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/CNNwithdist.png)

具体流程：对数据使用两个卷积核进行操作，得到**两个输出的feature map，分别作为真实输出的均值和标准差**。随后使用高斯分布从两个feature map中采样，得到该层feature map的激活值，作为下一层的输入。
激活值$\begin{aligned}b_j=A_i*\mu_i+\epsilon_j\odot\sqrt{A_i^2*(\alpha_i\odot\mu_i^2)}\end{aligned}$ (**Reparameterization**)
- 为了解决从分布中采样不可微的问题，使用基于重参数化(**Reparameterization**)的反向传播方法来估计梯度


### VGG16


[VGG16学习笔记 | 韩鼎の个人网站 (deanhan.com)](https://deanhan.com/2018/07/26/vgg16/)

### ResNet

[ResNet中的BasicBlock与bottleneck-CSDN博客](https://blog.csdn.net/sazass/article/details/116864275)

## RNN

相比一般的神经网络来说，他能够处理序列变化的数据

### LSTM

>[人人都能看懂的LSTM - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/32085405)
>讲的很好：[三分钟吃透RNN和LSTM神经网络](https://www.zhihu.com/tardis/zm/art/86006495?source_id=1003)


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240112195022.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20240112194924.png)

## U-Net

> [U-Net (labml.ai)](https://nn.labml.ai/unet/index.html)
> [anxingle/UNet-pytorch: medical image semantic segmentation](https://github.com/anxingle/UNet-pytorch)


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240528101406.png)


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


## GAN

> [生成对抗网络——原理解释和数学推导 - 黄钢的部落格|Canary Blog](https://alberthg.github.io/2018/05/05/introduction-gan/)

GAN在结构上受博弈论中的二人零和博弈 （即二人的利益之和为零，一方的所得正是另一方的所失）的启发，训练过程：
- 训练判别器：首先我们随机初始化生成器 G，并输入一组随机向量(Randomly sample a vactor)，以此产生一些图片，并把这些图片标注成 0（假图片）。同时把来自真实分布中的图片标注成 1（真图片）。两者同时丢进判别器 D 中，以此来训练判别器 D 。实现当输入是真图片的时候，判别器给出接近于 1 的分数，而输入假图片的时候，判别器给出接近于 0 的低分。
- 训练生成器：对于生成网络，目的是生成尽可能逼真的样本。所以在训练生成网络的时候，我们需要联合判别网络一起才能达到训练的目的。也就是说，通过将两者串接的方式来产生误差从而得以训练生成网络。步骤是：我们通过随机向量（噪声数据）经由生成网络产生一组假数据，并将这些假数据都标记为 1 。然后将这些假数据输入到判别网路里边，火眼金睛的判别器肯定会发现这些标榜为真实数据（标记为1）的输入都是假数据（给出低分），这样就产生了误差。在训练这个串接的网络的时候，一个很重要的操作就是不要让判别网络的参数发生变化，只是把误差一直传，传到生成网络那块后更新生成网络的参数。这样就完成了生成网络的训练了。

已知真实的分布$p_{data}(x)$，如何找到最合适的参数z，来使得生成的$p_{model}(x;z)$与真实分布之间的差异最小——极大似然估计：
- $\theta_{ML}=arg\operatorname*{\max}_{\theta}p_{model}(X;\theta)=arg\operatorname*{max}_{\theta}\prod_{i=1}^mp_{model}(x^{(i)};\theta)$
- $\theta_{ML}=arg\underset{\theta}{\operatorname*{max}}\sum_{i=1}^mlog\left.p_{model}(x^{(i)};\theta)\right.$ 通过log将累乘变成累加
- $\theta_{ML} = arg\max_{\theta} E_{x \sim \hat{p}_{data}} log p_{model}(x;\theta)$ 由于缩放代价函数不会影响求导和求argmax，因此除以m来将求和变成期望，当m-->$\infty$ 时，经验分布就会是真实数据的分布$\hat{p}_{data}\to p_{data}(x)$

通过$p_{model(x)}$与$p_{data}(x)$之间的差异衡量，来训练G和D:
$$\min_{G}\max_{D}V(G,D)=\mathbb{E}_{x\sim p_{data}(x)}[logD(\mathbf{x})]+\mathbb{E}_{z\sim p_{z}(z)}[log(1-D(G(\mathbf{z})))]$$
- $D^{*}=arg\max_{D}V(G,D)$ 生成器固定，判别器D判断的越好，V=1+1越大，D越差，V=0+0越小
- $G^{*}=arg\min_{G}\max_{D}V(G,D)$ 判别器固定，生成器G生成的越好，V=1+0越小


## Diffusion Model

> [2. 扩散概率模型（diffusion probabilistic models） — 张振虎的博客 张振虎 文档](https://www.zhangzhenhu.com/aigc/%E6%89%A9%E6%95%A3%E6%A6%82%E7%8E%87%E6%A8%A1%E5%9E%8B.html#diffusion-probabilistic-model)

$q(x_t|x_{t-1}) = \mathcal{N} (\sqrt{\alpha_t} \ x_{t-1}, (1- \alpha_t ) \textit{I} )$

$\begin{align}\begin{aligned}x_{t} &=\sqrt{\alpha_t} \ x_{t-1} + \mathcal{N} (0, (1- \alpha_t ) \textit{I} )\\&=\sqrt{\alpha_t} \ x_{t-1} +  \sqrt{1- \alpha_t } \ \epsilon \ \ \ ,\epsilon \sim \mathcal{N} (0, \textit{I} )\end{aligned}\end{align}$

随着t的增大，$\alpha_{t}$在逐渐变小。这是由于前期如果加的噪声太多，会使得数据扩展的太快（比如突变），使得逆向还原变得困难； 同样因为后期数据本身已经接近随机噪声数据了，后期如果加的噪声不够多，相当于变化幅度小，扩散的太慢，这会使得链路变长需要的事件变多。 我们希望扩散的前期慢一点，后期快一点

通过设定超参数$\alpha_{0:T}$可以看出**前向加噪**的过程是可以直接通过公式计算的，没有未知参数。

$$
\begin{align}\begin{aligned}x_t &= \sqrt{\alpha_t} \ x_{t-1} + \sqrt{1-\alpha_t} \ \epsilon_{t}\\&= \sqrt{\alpha_t} \left(   \sqrt{\alpha_{t-1}} \ x_{t-2} + \sqrt{1-\alpha_{t-1}} \ \epsilon_{t-1}  \right ) + \sqrt{1-\alpha_t} \ \epsilon_{t}\\&=  \sqrt{\alpha_t \alpha_{t-1} }  \ x_{t-2}
+ \underbrace{ \sqrt{\alpha_t - \alpha_t \alpha_{t-1} }\ \epsilon_{t-1} + \sqrt{1- \alpha_t} \ \epsilon_{t}
  }_{\text{两个相互独立的0均值的高斯分布相加}}\\
&=  \sqrt{\alpha_t \alpha_{t-1} }  \ x_{t-2}
+ \underbrace{  \sqrt{ \sqrt{\alpha_t - \alpha_t \alpha_{t-1} }^2 + \sqrt{1- \alpha_t}^2  } \ \epsilon
}_{\text{两个方差相加，用一个新的高斯分布代替}}\\&= \sqrt{\alpha_t \alpha_{t-1} }  \ x_{t-2} + \sqrt{1- \alpha_t \alpha_{t-1}} \ \epsilon\\&= ...\\&= \sqrt{\prod_{i=1}^t \alpha_i} \ x_0 + \sqrt{1- \prod_{i=1}^t \alpha_i }  \ \epsilon\\&= \sqrt{\bar{ \alpha}_t } \ x_0 + \sqrt{1- \bar{ \alpha}_t }  \ \epsilon  \ \ \ ,
\bar{\alpha} = \prod_{i=1}^t \alpha_i ,\ \ \epsilon \sim \mathcal{N}(0,\textit{I})\\&\sim \mathcal{N}(\sqrt{\bar{\alpha}_t } \ x_0,  (1- \bar{ \alpha}_t)    \textit{I})\end{aligned}\end{align}
$$

而逆向过程需要从噪声开始，逐步解码成一个有意义的数据 $p(x_{0:T}) = p(x_T) \prod_{t={T-1}}^0 p(x_{t}|x_{t+1})$
- 在这里$p(x_T) \sim \mathcal{N}(0,\textit{I})$ 是高斯分布，但是$p_{\theta}(x_{t}|x_{t+1})$ 是难以求解的(分母部分含有积分，且没有解析解)，依次使用网络来拟合学习这一条件概率分布 $p(x_{t}|x_{t+1})=\frac{p(x_{t},x_{t+1})}{p(x_{t+1})}=\frac{p(x_{t+1}|x_{t})p(x_{t})}{\int_{-\infty}^{+\infty}p(x_{t+1}|x_{t})p(x_{t})dx_{t}}.$
- 目标函数（ELBO）我们知道学习一个概率分布的未知参数的常用算法是极大似然估计， 极大似然估计是通过极大化观测数据的对数概率（似然）实现的

### DDPM(Denoising Diffusion Probabilistic Models)

$\text{Maximize E}_{q(x_1:x_T|x_0)}[log\left(\frac{P(x_0;x_T)}{q(x_1:x_T|x_0)}\right)]$

$q(x_t|x_0)$ 可以只做一次 sample(给定一系列 $\beta$)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023100329.png)

DDPM 的 Lower bound of $logP(x)$ 
复杂公式推导得到： ELBO函数来约束网络训练
$logP(x) \geq \operatorname{E}_{q(x_1|x_0)}[logP(x_0|x_1)]-KL\big(q(x_T|x_0)||P(x_T)\big)-\sum_{t=2}^{T}\mathrm{E}_{q(x_{t}|x_{0})}\bigl[KL\bigl(q(x_{t-1}|x_{t},x_{0})||P(x_{t-1}|x_{t})\bigr)\bigr]$
- $\mathbb{E}_{q(x_{1}|x_0)}\left[\ln p_{\theta}(x_0|x_1)\right]$ 重建项，从隐式变量中重建出原来的数据$x_{0}$
- $\mathbb{E}_{q(x_{T-1}|x_0)}\left[D_{KL}{q(x_T|x_{T-1})}{p(x_T)}\right]$ 最后一个数据是高斯噪声，因此当T足够大，这一项趋于0
- $\mathbb{E}_{q(x_{t-1}, x_{t+1}|x_0)}\left[D_{KL}{q(x_{t}|x_{t-1})}{p_{{\theta}}(x_{t}|x_{t+1})}\right]$ KL散度度量， consistency term。这一项用来最小化$q(x_{t}|x_{t-1})$ 与 $p_{{\theta}}(x_{t}|x_{t+1})$ 之间的差异。期望是关于两个变量的，用采样法（MCMC）同时对两个随机变量进行采样，会导致更大的方差，这会使得优化过程不稳定，不容易收敛，可以将$p_{{\theta}}(x_{t}|x_{t+1})$优化为：$q(x_{t-1}|x_t, x_0)$

$q(x_t | x_{t-1}, x_0) = \frac{q(x_{t-1} \mid x_t, x_0)q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}$

条件独立性：假设有三个随机变量A，B，C，三者依赖关系：A-->B-->C，当B值已知时，A和C相互独立，此时$P(C|B)=P(C|B,A)$ 成立  $q(x_t | x_{t-1}) = q(x_t | x_{t-1}, x_0)$
联合概率的基本性质：$q(x_t, x_{t-1}, x_0) = q(x_t \mid x_{t-1}, x_0) q(x_{t-1} \mid x_0) q(x_0)$ 另一种形式$q(x_t, x_{t-1}, x_0) = q(x_{t-1} \mid x_t, x_0) q(x_t \mid x_0) q(x_0)$

$q(x_t \mid x_{t-1}, x_0) q(x_{t-1} \mid x_0) q(x_0) = q(x_{t-1} \mid x_t, x_0) q(x_t \mid x_0) q(x_0)$
即得$q(x_t | x_{t-1}, x_0) = \frac{q(x_{t-1} \mid x_t, x_0)q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}$ 和 $q(x_{t-1}|x_{t},x_{0}) =\frac{q(x_{t}|x_{t-1})q(x_{t-1}|x_{0})}{q(x_{t}|x_{0})}$

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240923093718.png)


**直接预测原始样本** $x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon.$
- $q(x_{t-1}|x_{t},x_{0}) =\frac{q(x_{t}|x_{t-1})q(x_{t-1}|x_{0})}{q(x_{t}|x_{0})}$ 为一个 Gaussian distribution [推导过程](https://www.zhangzhenhu.com/aigc/%E6%89%A9%E6%95%A3%E6%A6%82%E7%8E%87%E6%A8%A1%E5%9E%8B.html#equation-eq-ddpm-036)
  - 均值+方差： $mean = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}x_{0}+\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})x_{t}}{1-\bar{\alpha}_{t}}$ ，$variance = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta$

$x_0$需要通过网络预测来获得，因此：
$\mu_q(x_t, x_0) = { \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)x_0}{1 -\bar\alpha_{t}}}$
$\mu_{\theta}={\mu}_{{\theta}}(x_t, t) = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)\hat x_{{\theta}}(x_t, t)}{1 -\bar\alpha_{t}}$ 网络预测$\hat{x}_{\theta}$ ，然后这个分布，采样得到$x_{t-1}$

**然而直接预测$x_{0}$时非常困难得，每个步骤都只给定每一步的t来预测同样的输出，(trained)相同的网络参数很难完成这样的预测**

因此DDPM转换思路来**预测每步t中添加的噪声**：
$x_0 =  \frac{x_t -\sqrt{1- \bar{ \alpha}_t }  \ \epsilon_t }{ \sqrt{\bar{\alpha}_t }  },\ \ \epsilon_t \sim \mathcal{N}(0,\textit{I})$

均值：$\mu_{\theta}={\mu}_{{\theta}}(x_t, t) = \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}\sqrt{\alpha_t}} {\hat\epsilon}_{ {\theta}}(x_t, t)$ [推导过程](https://www.zhangzhenhu.com/aigc/%E6%89%A9%E6%95%A3%E6%A6%82%E7%8E%87%E6%A8%A1%E5%9E%8B.html#equation-eq-ddpm-054) (将$x_{0}$重写成$\epsilon$)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022210535.png)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022210617.png)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231023102747.png)


### DDPM 三种形式

逆向生成过程：$q(x_{t-1}|x_t,x_0) \sim \mathcal{N}(x_{t-1},\mu_q,\Sigma_{q(t)})$

方差：
$\Sigma_q(t) = \frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{ 1 -\bar\alpha_{t}}  \textit{I} = \sigma_q^2(t)   \textit{I}$


#### 1）直接预测初始样本
$\mu_q(x_t, x_0) = { \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)x_0}{1 -\bar\alpha_{t}}}$

$\mu_{\theta}={\mu}_{{\theta}}(x_t, t) = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})x_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)\hat x_{{\theta}}(x_t, t)}{1 -\bar\alpha_{t}}$

#### 2）预测噪声
$\mu_q(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}\sqrt{\alpha_t}}\ \epsilon$

$\mu_{\theta}={\mu}_{{\theta}}(x_t, t) = \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}\sqrt{\alpha_t}} {\hat\epsilon}_{ {\theta}}(x_t, t)$

#### 3）预测分数
${\mu}_q(x_t, x_0) =  \frac{1}{\sqrt{\alpha_t}}x_t + \frac{1 - \alpha_t}{\sqrt{\alpha_t}}\nabla\log p(x_t)$

${\mu}_q(x_t, x_0) =  \frac{1}{\sqrt{\alpha_t}}x_t + \frac{1 - \alpha_t}{\sqrt{\alpha_t}} s_{\theta}(x_t,t)$


### DDPM Code

[mikonvergence/DiffusionFastForward: DiffusionFastForward: a free course and experimental framework for diffusion-based generative models (github.com)](https://github.com/mikonvergence/DiffusionFastForward)

#### Schedule

* `betas`: $\beta_t$ , `betas=torch.linspace(1e-4,2e-2,num_timesteps)`
* `alphas`: $\alpha_t=1-\beta_t$ 
* `alphas_sqrt`:  $\sqrt{\alpha_t}$
* `alphas_prod`: $\bar{\alpha}_t=\prod_{i=0}^{t}\alpha_i$
* `alphas_prod_sqrt`: $\sqrt{\bar{\alpha}_t}$

diffusion step 0,1,...,t,...T：随着t增大，beta应该逐渐增大
- cosine: 
  - $\alpha_{cumprod} =\cos\left( \left( \frac{\frac{t}{T}+s}{(1+s)}\cdot \pi \cdot 0.5 \right)^{2} \right)$ ，value: $\cos\left( \frac{s}{1+s} \cdot \frac{\pi}{2} \right)$ ~$\cos\left( \frac{\pi}{2} \right)$ = $\cos(0)$ ~ $\cos\left( \frac{\pi}{2} \right)$ if s --> 0
  - $\alpha_{cumprod} = \frac{\alpha_{cumprod}}{\alpha_{cumprod}^{max}}$, 让最大的$\alpha$不超过1
  - $\beta$ = `1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])`
- linear: 
  - $linspace\left( scale*0.0001,scale*0.02,T \right)$ , $scale=\frac{1000}{T}$

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240924095523.png)


```python
# consine beta
def cosine_beta_schedule(timesteps, s=0.008):
  """
  cosine schedule
  as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
  """
  steps = timesteps + 1
  x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
  alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
  alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
  betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
  return torch.clip(betas, 0, 0.999)

# linear beta
def linear_beta_schedule(timesteps):
  scale = 1000 / timesteps
  beta_start = scale * 0.0001
  beta_end = scale * 0.02
  return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
```

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

### Why Does Diffusion Work Better than Auto-Regression?

> [Why Does Diffusion Work Better than Auto-Regression? - YouTube](https://www.youtube.com/watch?v=zc5NTeJbk-k)

- 分类任务：图片-->类别

如何生成图片(Thinking)：
- 从任意数据中预测整张图片(真实图片做标签)，这样训练集标签图片的mean value会变blurring(**blurry mess**)。(分类任务中训练集标签01的meaning value不会收到很大影响)
- **Auto-Regressor**: 正向过程是不断擦除图像，网络被训练为undo这个过程，即不断预测图片的下一个像素是什么颜色(like ChatGPT)
  - 考虑来预测图片中的单个像素，这样训练集标签的mean value是另一个颜色值(网络根据输入的图片来预测单个颜色值)。每个像素的颜色训练一个网络来进行预测，通过多个网络，依次预测每个像素值，最终得到整张图片。这样就能生成plausible(似乎是真的) image，out of nothing(凭空)。**然而每次生成的图片是相同的**
  - 考虑添加随机，之前predicted pixel是概率分布中概率最大的那个颜色值，**不去这样选择而是随机选择某个概率的颜色作为本次的预测**
  - 缺点是随着像素量的增大，计算量非常大，要生成一张大像素图像是非常耗时的。当然可以造数据集时一次remove多个像素，在训练中一次预测多个像素。**但是不能过多，这样依然会造成blurry mess** Trade-off：更快但是blurry mess，更慢但是更准确

**Blurry mess**：
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240808183321.png)

**Why predicting one pixel is work**： 
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240808183400.png)

该问题只会出现在预测的值是互相关的情况(在图像中相近的像素通常是强相关的，这在按顺序remove时出现)，假设预测的值相互独立的情况(随机remove像素)。**remove像素可以从另一种角度进行实现，即给每个像素添加噪声**

### Stable Diffusion

> [Stable Diffusion 图片生成原理简述 « bang’s blog](https://blog.cnbang.net/tech/3766/)

目前常见的 UI 有 WebUI 和 ComfyUI

#### 模型

模型格式：
- 主模型 checkpoints：*ckpt, safetensors*
- 微调模型
  - LoRA 和 LyCORIS 控制画风和角色：*safetensors*
  - 文本编码器模型：*pt,safetensors*
    - Embedding 输入文本 prompt 进行编码 *pt*
  - Hypernetworks 低配版的 lora *pt*
  - ControlNet
  - VAE 图片与潜在空间 *pt*


#### 采样器

[Stable diffusion采样器全解析，30种采样算法教程](https://www.bilibili.com/video/BV1FN411i7sB/)
DPM++2M Karras，收敛+速度快+质量 OK

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231022195103.png)


## MLP

### 前向传播

根据每层的输入、权重weight和偏置bias，求出该层的输出，然后经过激活函数。按此一层一层传递，最终获得输出层的输出。

### 反向传播

>[神经网络之反向传播算法（BP）公式推导（超详细） - jsfantasy - 博客园 (cnblogs.com)](https://www.cnblogs.com/jsfantasy/p/12177275.html)
>[ML Lecture 7: Backpropagation - YouTube](https://www.youtube.com/watch?v=ibJpTrp5mcE)

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



## Transformer

> [The Illustrated Transformer – Jay Alammar – Visualizing machine learning one concept at a time.](https://jalammar.github.io/illustrated-transformer/)
> [The Transformer Family | Lil'Log](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/)
> [注意力机制的本质|Self-Attention|Transformer|QKV矩阵_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1dt4y1J7ov/?spm_id_from=333.337.search-card.all.click&vd_source=1dba7493016a36a32b27a14ed2891088) 


输入X，通过三个不同的权重
$f(X)=softmax(XW_QX^TW_K/\sqrt{d})XW_V$  OR $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\frac{\mathbf{Q} {\mathbf{K}}^\top}{\sqrt{d_k}})\mathbf{V}$
- $a_{ij} = \text{softmax}(\frac{\mathbf{q}_i {\mathbf{k}_j}^\top}{\sqrt{d_k}})= \frac{\exp(\mathbf{q}_i {\mathbf{k}_j}^\top)}{ \sqrt{d_k} \sum_{r \in S_i} \exp(\mathbf{q}_i {\mathbf{k}_r}^\top) }$
- 其中$\sqrt{d}$是为了防止维度过高导致的梯度消失，d是数据的维度

**softmax(Q与K的乘积)** 可以看作权重，通过权重来插值V得到最终的输出。通过使Q与K距离越小时，对
应的权重更大，即此时Key对应的Value对输出的贡献更大，让网络注意到这个贡献更大的Key

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240923164610.png)
![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240923164633.png)


一个$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\frac{\mathbf{Q} {\mathbf{K}}^\top}{\sqrt{d_k}})\mathbf{V}$计算为Attention的一个Head，多头注意力就是多个Head：
- $\begin{aligned}\text{MultiHeadAttention}(\mathbf{X}_q, \mathbf{X}_k, \mathbf{X}_v) &= [\text{head}_1; \dots; \text{head}_h] \mathbf{W}^o \\ \text{where head}_i &= \text{Attention}(\mathbf{X}_q\mathbf{W}^q_i, \mathbf{X}_k\mathbf{W}^k_i, \mathbf{X}_v\mathbf{W}^v_i)\end{aligned}$
- $\mathbf{W}^o \in \mathbb{R}^{d_v \times d}$ 为输出的线性transformation

vanilla Transformer model:
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240923165513.png)


Positiolal Encoding

$$\text{PE}(i,\delta) = 
\begin{cases}
\sin(\frac{i}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta'\\
\cos(\frac{i}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta' + 1\\
\end{cases}$$
- the token position $i=1,\dots,L$
- the dimension $\delta=1,\dots,d$

_Learned positional encoding_, as its name suggested, assigns each element with a learned column vector which encodes its _absolute_ position
[Convolutional Sequence to Sequence Learning | Abstract](https://arxiv.org/abs/1705.03122)

## INN

[Interval Neural Networks 2020](https://arxiv.org/pdf/2003.11566) 通过物理参数和measured response来辨识 集中负载

$y_{k}^{I}=f(x_{1},x_{2},\cdots,x_{l})=[\underline{y}_{k},\overline{y}_{k}]=f\left(\sum_{j=1}^{m}\left[\underline{v}_{jk},\overline{v}_{jk}\right]u_{j}-[\underline{\lambda}_{k},\overline{\lambda}_{k}]\right)$

![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240920162900.png)![image.png|333](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240920163041.png)

INN 与 MLP 两者预测区间参数的区别
INN offers greater fitting flexibility due to the interval of weight and bias

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240920163527.png)


## KAN

[KindXiaoming/pykan: Kolmogorov Arnold Networks](https://github.com/KindXiaoming/pykan?tab=readme-ov-file)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240505163014.png)


## SNNs

[第三代神经网络初探：脉冲神经网络（Spiking Neural Networks） - 知乎](https://zhuanlan.zhihu.com/p/531524477)




# 其他概念


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

# 强化学习

> [蘑菇书EasyRL](https://datawhalechina.github.io/easy-rl/#/)

## Paper

> [GameNGen](https://gamengen.github.io/)
