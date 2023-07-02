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

# GPU

```
Neus: 
torch.set_default_tensor_type('torch.cuda.FloatTensor')
parser.add_argument('--gpu', type=int, default=0)
torch.cuda.set_device(args.gpu)

self.device = torch.device('cuda')
network = Network(**self.conf['model.nerf']).to(self.device)

#################################################################
NeRF:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeRF().to(device)
render_poses = torch.Tensor(render_poses).to(device)
```

```
torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
如果有多个GPU，我们使用`torch.device(f'cuda:{i}')` 来表示第i块GPU（i从0开始）。 另外，`cuda:0`和`cuda`是等价的。

查询gpu数量
torch.cuda.device_count()

查询张量所在设备
x = torch.tensor([1, 2, 3])
x.device   #device(type='cpu') 默认为gpu，也可为cpu
```

两张量相互运算需要在同一台设备上`Z = X.cuda(1)`

![image.png](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230702204507.png)

```
给网络指定设备
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
```

==只要所有的数据和参数都在同一个设备上， 我们就可以有效地学习模型==

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