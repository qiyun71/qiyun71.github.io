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
# 环境配置

## windows 

>[关于国内conda安装cuda11.6+pytorch的那些事。 – 王大神 (dashen.wang)](https://dashen.wang/1283.html)

使用miniconda创建虚拟环境
- conda create -n mine python=3.8
- conda activate mine

安装cuda

```
换源：
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls true

安装：
conda install pytorch torchvision torchaudio pytorch-cuda=11.6

Found conflicts:
Package pytorch conflicts for:  
torchaudio -> pytorch[version='1.10.0|1.10.1|1.10.2|1.11.0|1.12.0|1.12.1|1.13.0|1.13.1|2.0  
.0|2.0.1|1.9.1|1.9.0|1.8.1|1.8.0|1.7.1|1.7.0|1.6.0']  
torchvision -> pytorch[version='1.10.0|1.10.1|1.10.2|2.0.1|2.0.0|1.13.1|1.13.0|1.12.1|1.12  
.0|1.11.0|1.9.1|1.9.0|1.8.1|1.8.0|1.7.1|1.7.0|1.6.0|1.5.1']
...

使用以下命令安装
> conda install -c gpytorch gpytorch

安装带cuda的torch
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 --user
```

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

# 私有成员
带双下划线函数

| function   | brief description |
| ---------- | ----------------- |
| `nn.module.__repr__`  |        当print(model)时会运行该函数           |
| `__del__`           |         当`del object`时运行该函数           |


# torch.cuda

## cuda事件计算程序运行时间

```python
iter_start = torch.cuda.Event(enable_timing = True)
iter_end = torch.cuda.Event(enable_timing = True)
iter_start.record()
# iter 1 code
iter_end.record()

print(f'iter time: {iter_start.elapsed_time(iter_end)}')
```

eg:
```python
import torch

iter_start = torch.cuda.Event(enable_timing = True)
iter_end = torch.cuda.Event(enable_timing = True)
iter_start.record()

a = torch.tensor([1,2,3,4,5,6,7,8,9,10]).cuda()

iter_end.record()

timestamp = iter_start.elapsed_time(iter_end)
print(f'iter time: {timestamp:03f}')
```