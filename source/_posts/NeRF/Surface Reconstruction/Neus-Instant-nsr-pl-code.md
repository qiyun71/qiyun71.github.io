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

文件结构：

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