---
title: Sampling
date: 2023-07-29 16:51:42
tags:
  - NeRF
  - Sampling
categories: 3DReconstruction/Basic Knowledge/NeRF/NeRF
---

从相机原点出发，通过像素点射出一条光线，在光线上进行采样

<!-- more -->

# 直线光线采样

将像素看成一个点，射出的光线是一条直线

![Network.png|666](https://raw.githubusercontent.com/yq010105/Blog_images/main/Network.png)
(NerfAcc)可大致分为：
- 平均采样(粗采样)
- 空间跳跃采样(NGP中对空气跳过采样)
- 逆变换采样(根据粗采样得到的w分布进行精采样)

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230711125031.png)
## 平均采样

i.e. 粗采样，在光线上平均采样n个点

## 占据采样

Occupancy Grids
通过在某分辨率占用网格中进行更新占用网格的权重，来确定哪些网格需要采样

## 逆变换采样

### NeRF

简单的逆变换采样方法：根据粗采样得到的权重进行逆变换采样，获取精采样点
[逆变换采样 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/80726483)

```python
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans # weights : chunk * 62

    # 归一化weights
    pdf = weights / torch.sum(weights, -1, keepdim=True) # pdf : chunk * 62
    cdf = torch.cumsum(pdf, -1) # cdf : chunk * 62
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))  = (chunk, 63)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples]) # u : chunk * N_samples
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples] # new_shape : chunk * N_samples
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape) # u : chunk * N_samples
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous() # 确保张量在内存中是连续存储的
    # inds : chunk * N_samples
    inds = torch.searchsorted(cdf, u, right=True) # 将u中的元素在cdf中进行二分查找，返回其索引
    below = torch.max(torch.zeros_like(inds-1), inds-1) # below : chunk * N_samples
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds) # above : chunk * N_samples
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]  # (batch, N_samples, 63)
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g) 
    # unsqueeze(1) : (batch, 1, 63)
    # expand : (batch, N_samples, 63)
    # cdf_g : (batch, N_samples, 2)    
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0]) # denom : chunk * N_samples
    # 如果denom小于1e-5，就用1代替
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples # samples : chunk * N_samples
```

### Mip-NeRF360

构建了一个提议网格获取权重来进行精采样（下）

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230722152752.png)



# 锥形光线采样

## Mip-NeRF

将像素看成有面积的圆盘，射出的光线为一个圆锥体
- 使用多元高斯分布来近似截锥体


![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230721125154.png)


## Tri-MipRF

- 使用一个各项同性的球来近似截锥体


![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230726164225.png)

## Zip-NeRF
Multisampling

多采样：在一个截锥体中沿着光线采样6个点，每个点之间旋转一个角度

![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20230729141223.png)


# 混合采样

NerfAcc：占据+逆变换采样
先使用占据网格确定哪些区域需要采样，再通过粗采样得到的权重使用逆变换采样进行精采样得到采样点