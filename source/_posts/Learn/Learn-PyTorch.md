---
title: PyTorch代码学习笔记
date: 2023-06-21 12:26:14
tags:
    - Python
    - Pytorch
categories: Learn
---

基于Pytorch学习DL时，学习到的一些技巧/code

<!-- more -->

# 优化器

## Adam多个model参数，然后更新lr

Adam_in_Neus: params_to_train is a list

```
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device) # 创建一个NeRF网络
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device) # 创建一个SDF网络
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
```

然后更新学习率

`g = self.optimizer.param_groups[index]`

```
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor
```
**from**
```
    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor
```