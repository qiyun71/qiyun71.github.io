---
title: Time Series Generation
date: 2024-09-20 22:33:08
tags:
  - 
categories: Blog&Book&Paper/Read/Papers
---

[Time Series Generation | Papers With Code](https://paperswithcode.com/task/time-series-generation/latest)


<!-- more -->

# Diffusion

## [Diffusion TS](https://openreview.net/pdf?id=4h1apFjO99) 2024

>[Y-debug-sys/Diffusion-TS: [ICLR 2024] Official Implementation of "Diffusion-TS: Interpretable Diffusion for General Time Series Generation"](https://github.com/y-debug-sys/diffusion-ts) code

时间序列数据可分为 趋势项Trend+周期项Seasonality+误差项Error
$x_j=\zeta_j+\sum_{i=1}^ms_{i,j}+e_j,\quad j=0,1,\ldots,\tau-1,$


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240920223417.png)


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



# VAE

## [TimeVAE](https://arxiv.org/pdf/2111.08095v3)

>[abudesai/timeVAE: TimeVAE implementation in keras/tensorflow](https://github.com/abudesai/timeVAE)
>[abudesai/syntheticdatagen: synthetic data generation of time-series data](https://github.com/abudesai/syntheticdatagen)


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240920224735.png)

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240920224814.png)
