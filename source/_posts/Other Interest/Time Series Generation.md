---
title: Time Series Generation (Uncertainty)
date: 2024-09-20 22:33:08
tags: 
categories: Other Interest
---

[Time Series Generation | Papers With Code](https://paperswithcode.com/task/time-series-generation/latest)
- 生成样本，解决样本不足的问题

<!-- more -->

# Review

## [GAN](https://dl.acm.org/doi/pdf/10.1145/3559540)

>[Generative Adversarial Networks in Time Series: A Systematic Literature Review](https://dl.acm.org/doi/pdf/10.1145/3559540)

- Autoregressive (AR) to forecast data point of time series：$x_{t+1} = c + \theta_1x_t + \theta_2x_{t-1} + \epsilon$。然而其内在是确定性的由于没有在未来系统状态的计算中引入随机性
- Generative Methods：
  - Autoencoder (AE)
  - variational autoencoders (VAEs)
  - recurrent neural network (RNN)
  - GAN is inherent instability, suffer from issues: 
    - non-convergence, A non-converging model does not stabilize and continuously oscillates, causing it to diverge. 
    - diminishing/vanishing gradients, Diminishing gradients prevent the generator from learning anything, as the discriminator becomes too successful. 
    - mode collapse. Mode collapse is when the generator collapses, producing only uniform samples with little to no variety.
    - 没有比较好的评价指标

GAN Model task: 生成器G要最大化判别器D的错误率，而判别器D则是要最小化其错误率
$$\min_{G}\max_{D}V(G,D)=\mathbb{E}_{x\sim p_{data}(x)}[logD(\mathbf{x})]+\mathbb{E}_{z\sim p_{z}(z)}[log(1-D(G(\mathbf{z})))]$$

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240924163352.png)

应用：
- Data Augumentation
- Imputation
- Denoising
- Anomaly Detection异常检测

# Diffusion

## [Diffusion TS](https://openreview.net/pdf?id=4h1apFjO99) 2024

Diffusion + Transformer

UNCONDITIONAL TIME SERIES GENERATION，输入的是噪声数据 `img = torch.randn(shape, device=device)`
CONDITIONAL TIME SERIES GENERATION：这里的条件代表的也是time series(缺失部分数据)
- Imputation 补全
- Forecasting 预测

>[Y-debug-sys/Diffusion-TS: [ICLR 2024] Official Implementation of "Diffusion-TS: Interpretable Diffusion for General Time Series Generation"]( https://github.com/y-debug-sys/diffusion-ts ) code

时间序列数据可分为: 
- 趋势项 Trend+周期项 Seasonality+误差项 Error
- $x_j=\zeta_j+\sum_{i=1}^ms_{i,j}+e_j,\quad j=0,1,\ldots,\tau-1,$
- $\hat{x}_0(x_t,t,\theta)=V_{tr}^t+\sum_{i=1}^DS_{i,t}+R$

Trend: $V_{tr}^t=\sum_{i=1}^D(\boldsymbol{C}\cdot\mathrm{Linear}(w_{tr}^{i,t})+\mathcal{X}_{tr}^{i,t}),\quad\boldsymbol{C}=[1,c,\ldots,c^p]$
- D 代表由 D 个 Decoder block
- $\mathcal{X}_{tr}^{i,t}$ 是 i-th 个 Decoder block 输出的均值，*趋势项加上这个可以让其在y轴上下移动*，该项的累加在代码中并不是简单的求和， 而是通过一个conv1d来动态调整求和时的权重，此外还会加上最后一个decoder block输出的均值
- C is the matrix of powers of vector $c=\frac{[0,1,2,\dots,\tau-2,\tau-1]^{T}}{\tau}$ $\tau$ is the time series length
  - $y=a_{0}+a_{1}x+a_{2}x^{2}+a_{3}x^{3}+\dots+a_{p}x^{p}$ , 
    - 系数a即网络的输出$\mathrm{Linear}(w_{tr}^{i,t})$
    -  $x,x^{2},x^{3}\dots,x^{p}$  **本质上是通过C来表示**
- P is a small degree (p=3) to model low frequency behavior  p=3代表用3阶来拟合趋势项已经足够

TrendBlock:
- Input: input $w_{tr}^{i,t}$ 
- Output: trend_vals $\boldsymbol{C}\cdot\mathrm{Linear}(w_{tr}^{i,t})$

```python
###### TrendBlock ######
# forward:
b, c, h = input.shape # (batch_size, seq_len, d_model)
x = self.trend(input).transpose(1, 2) # (batch_size, trend_poly, out_feat)
trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
trend_vals = trend_vals.transpose(1, 2) # (batch_size, seq_len, out_feat)

# __init__: 
trend_poly = 3
self.trend = nn.Sequential(
    nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
    act,
    Transpose(shape=(1, 2)),
    nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
)
lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1) # out_dim = seq_len = time series length
self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0) # trend_poly, out_dim
```

Seasonality: $S_{i,t}=\sum_{k=1}^{K}A_{i,t}^{\kappa_{i,t}^{(k)}}\left[\cos(2\pi f_{\kappa_{i,t}^{(k)}}\tau c+\Phi_{i,t}^{\kappa_{i,t}^{(k)}})+\cos(2\pi\bar{f}_{\kappa_{i,t}^{(k)}}\tau c+\bar{\Phi}_{i,t}^{\kappa_{i,t}^{(k)}})\right],$  
- Amplitude $A_{i,t}^{(k)}=\left|\mathcal{F}(w_{seas}^{i,t})_{k}\right|$
- Phase $\Phi_{i,t}^{(k)}=\phi\left(\mathcal{F}(w_{seas}^{i,t})_{k}\right)$
- $\mathcal{F}$ is the discrete Fourier transform
- $\kappa_{i,t}^{(1)},\cdots,\kappa_{i,t}^{(K)}=\arg\text{TopK}\\k\in\{1,\cdots,\lfloor\tau/2\rfloor+1\}$ 选择 top K 的幅值和相位，K is a hyperparameter
- $f_{k}$ represents the Fourier frequency of the corresponding index k, $\bar{f}_{k}$ 为 $f_{k}$ 的共轭
- $\tau c=\tau \cdot\frac{[0,1,2,\dots,\tau-2,\tau-1]^{T}}{\tau}=[0,1,2,\dots,\tau-2,\tau-1]^{T}$

FourierLayer:
- Input: x $w_{seas}^{i,t}$ 
- Output:  x_time $S_{i,t}$ 
  - `x_freq, x_freq.conj()` 是$\mathcal{F}(w_{seas}^{i,t})_{k}$ 和其共轭，通过abs 和 angle 得到幅值和相位
  - `f, -f` is $f_{k}$ and $\bar{f}_{k}$

```python
###### FourierLayer ######
from einops import rearrange, reduce, repeat

# forward:
b, t, d = x.shape
x_freq = torch.fft.rfft(x, dim=1)  # (b, t, d) -> (b, t//2+1, d)

if t % 2 == 0:
    x_freq = x_freq[:, self.low_freq:-1]
    f = torch.fft.rfftfreq(t)[self.low_freq:-1]
else:
    x_freq = x_freq[:, self.low_freq:]
    f = torch.fft.rfftfreq(t)[self.low_freq:]
# print(x_freq.shape) # (b, t//2 - sel.low_freq, d)
# print(f.shape) # (t//2 - sel.low_freq)
x_freq, index_tuple = self.topk_freq(x_freq) # (b, top_k, d)
f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device) # (b, t//2 - sel.low_freq, d)
f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device) # (b, top_k, 1, d)
x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
f = torch.cat([f, -f], dim=1)
t = rearrange(torch.arange(t, dtype=torch.float),
              't -> () () t ()').to(x_freq.device)
amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
x_time = amp * torch.cos(2 * math.pi * f * t + phase)
return reduce(x_time, 'b f t d -> b t d', 'sum')

# __init__ and function:
self.d_model = d_model
self.factor = factor
self.low_freq = low_freq

def topk_freq(self, x_freq):
  length = x_freq.shape[1]
  top_k = int(self.factor * math.log(length))
  # print(top_k)
  values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True) # (b, top_k, d)
  # print(values.shape, indices.shape); print(values); print(indices)
  mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij') # (b, d)
  index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1)) # (b, 1, d) (b, top_k, d) (b, 1, d)
  x_freq = x_freq[index_tuple] # (b, top_k, d) 🤔高级索引方法(元组)
  return x_freq, index_tuple
```

Error: R is the output of the last decoder block, which can be regarded as the sum of residual
periodicity and other noise. 代码中的R为最后一个decoder block输出减去其平均值，并将平均值加到趋势项中，让周期项不包含任何的y轴方向信息

时间序列数据被表示为 $\hat{x}_0(x_t,t,\theta)=V_{tr}^t+\sum_{i=1}^DS_{i,t}+R$，然后通过DDPM来直接预测原始样本，反向生成过程被表示为：$x_{t-1}=\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\hat{x}_0(x_t,t,\theta)+\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t+\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_tz_t,$ 
- $z_t\sim\mathcal{N}(0,\mathbf{I}), \alpha_t=1-\beta_t\text{ and }\bar{\alpha}_t=\prod_{s=1}^t\alpha_s$

损失函数：
时域+FFT频域监督：$\mathcal{L}_\theta=\mathbb{E}_{t,x_0}\left[w_t\left[\lambda_1\|x_0-\hat{x}_0(x_t,t,\theta)\|^2+\lambda_2\|\mathcal{F}\mathcal{FT}(x_0)-\mathcal{FFT}(\hat{x}_0(x_t,t,\theta))\|^2\right]\right]$
- $\mathcal{L}_{simple}=\mathbb{E}_{t,x_{0}}\left[w_{t}\|x_{0}-\hat{x}_{0}(x_{t},t,\theta)\|^{2}\right],\quad w_{t}=\frac{\lambda\alpha_{t}(1-\bar{\alpha}_{t})}{\beta_{t}^{2}}$
  - 其中权重被设置为当在 small t 时权重变小，让网络集中在更大的 diffusion step t(噪声较多的时候)
  - 由 $x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon.$，随着t的增大，$\alpha_{t}$ 在逐渐变小

上述过程为无条件 time series generation，本文还描述了conditional extensions of Diffusion-TS，即the modeled $x_{0}$ 以 targets y 为条件。

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240923180228.png)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240920223417.png)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240921202408.png)


```bash
# 环境配置
conda create -n Diffusion-TS
conda activate Diffusion-TS

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

```bash
# train with stocks dataset with gpu 0
python main.py --name stocks_exp1 --config_file ./Config/stocks.yaml --gpu 0 --train
```

# GANs

## [RGANs](https://arxiv.org/pdf/1706.02633v2) Most⭐2017

> [ratschlab/RGAN: Recurrent (conditional) generative adversarial networks for generating real-valued time series data.](https://github.com/ratschlab/RGAN) code

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240920224613.png)



## [SigCGANs](https://arxiv.org/pdf/2006.05421v2)

> [SigCGANs/Conditional-Sig-Wasserstein-GANs](https://github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs) code


## [TimeGAN](https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240923145250.png)

# VAE

## [TimeVAE](https://arxiv.org/pdf/2111.08095v3)

> [abudesai/timeVAE: TimeVAE implementation in keras/tensorflow](https://github.com/abudesai/timeVAE)
> [abudesai/syntheticdatagen: synthetic data generation of time-series data](https://github.com/abudesai/syntheticdatagen)


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240920224735.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240920224814.png)
