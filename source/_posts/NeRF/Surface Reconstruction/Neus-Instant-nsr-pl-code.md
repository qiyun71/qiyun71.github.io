---
title: Instant Neus的代码理解
date: 2023-07-03 22:02:46
tags:
    - Python
    - Code
    - Training & Inference Efficiency
categories: NeRF/Surface Reconstruction
---

Instant Neus的代码理解

>[Welcome to ⚡ PyTorch Lightning — PyTorch Lightning 1.9.5 documentation](https://lightning.ai/docs/pytorch/1.9.5/)

<!-- more -->

使用了PyTorch Lightning库
# 文件结构：

```
├───configs  # 配置文件
│ nerf-blender.yaml  
│ nerf-colmap.yaml  
│ neus-blender.yaml  
│ neus-bmvs.yaml  
│ neus-colmap.yaml  
│ neus-dtu.yaml  
│  
├───datasets  # 数据集加载
│ blender.py  
│ colmap.py  
│  colmap_utils.py  
│  dtu.py  
│  utils.py  
│  __init__.py  
│  
├───models  # model的神经网络结构和model的运算
│  base.py  
│  geometry.py  
│  nerf.py  
│  network_utils.py  
│  neus.py  
│  ray_utils.py  
│  texture.py  
│  utils.py  
│  __init__.py  
│  
├───scripts  # 自定义数据集时gen_poses+run_colmap生成三个bin文件['cameras', 'images', 'points3D']
│ imgs2poses.py  
│  
├───systems  # model模型加载和训练时每步的操作
│  base.py  
│  criterions.py  
│  nerf.py  
│  neus.py  
│  utils.py  
│  __init__.py  
│  
└───utils  
│ callbacks.py  
│ loggers.py  
│ misc.py  
│ mixins.py  
│ obj.py  
│ __init__.py  
```

# datasets

## init

```
datasets = {}

def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator

def make(name, config): # dtu ,config.datasets
    dataset = datasets[name](config) # dataset = datasets['dtu'](config)
    return dataset

from . import blender, colmap, dtu
```

## dtu

### load_K_Rt_from_P

### create_spheric_poses

### DTUDatasetBase

### DTUDataset

### DTUIterableDataset

### DTUDataModule
@datasets.register('dtu')

# models

## init

`@models.register('neus')` 修饰器的作用：
- 主要是为了实例化NeuSModel()的同时，在models字典中同时存入一个NeuSModel()值，对应的key为'neus'

当运行 `neus_model = NeuSModel()` 时，会运行`neus_model = register('neus')(NeusModel)`
返回给neus_model的值为decorator(cls) 函数的返回值，即NeusModel


```
models = {}

def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

def make(name, config):
    model = models[name](config)
    return model

from . import nerf, neus, geometry, texture
```

## base

### BaseModel

其他model需要继承于BaseModel

## neus

Neus中的两个网络

### VarianceNetwork

sigmoid函数的s参数在训练中变化

### NeuSModel
@models.register('neus')

#### setup

#### update_step
#### isosurface

#### get_alpha
#### forward_bg_
#### forward_
#### forward
#### train
#### eval
#### regularizations

#### export
@torch.no_grad()

## network_utils

各种编码方式和NeRF的MLP网络

### VanillaFrequency

Vanilla：最初始的
即NeRF中的频率编码方式

### ProgressiveBandHashGrid

### CompositeEncoding

### get_encoding

### VanillaMLP

NeRF中的MLP


### sphere_init_tcnn_network

### get_mlp


### EncodingWithNetwork

### get_encoding_with_network


## geometry

### contract_to_unisphere

### MarchingCubeHelper

### BaseImplicitGeometry

### VolumeDensity
@models.register('volume-density')

### VolumeSDF
@models.register('volume-sdf')

# systems

## init

```
systems = {}

def register(name):
    def decorator(cls):
        systems[name] = cls
        return cls
    return decorator

def make(name, config, load_from_checkpoint=None):
    if load_from_checkpoint is None:
        system = systems[name](config)
    else:
        system = systems[name].load_from_checkpoint(load_from_checkpoint, strict=False, config=config)
    return system

from . import nerf, neus
```

## neus

### NeuSSystem
@systems.register('neus-system')

#### prepare

#### forward
#### preprocess_data
#### training_step
#### validation_step
#### validation_epoch_end
#### test_step
#### test_epoch_end
#### export




