---
title: 基于Instant-nsr-pl创建一个项目
date: 2023-07-06 21:17:54
tags:
    - NeRF
    - MyProject
categories: NeRF
---

自己的项目：基于Instant-nsr-pl——[yq010105/NeRF-Mine (github.com)](https://github.com/yq010105/NeRF-Mine)

<!-- more -->

pip install -r requirements

tensorboard --port 6007 --logdir /root/tf-logs

文件结构：
- confs/ 配置文件
- encoder/ 编码方式
- process_data/ 处理数据集
- models/ 放一些网络的结构和网络的运行和方法
- systems/ 训练的程序
- utils/ 工具类函数
- run.py

- inputs/ 数据集
- outputs/ 输出和log文件
    - logs filepath: /root/tf-logs/name_in_conf/trial_name

```bash
# train
python run.py --config ./confs/dtu.yaml --train
```

# 代码结构

## confs

```
eg: dtu.yaml


'trainer': {'epochs': 200, 'val_freq': 5, 'outputs_dir': './outputs\\neu  
s-dtu-Miku', 'trial_name': '@20230723-131350', 'save_dir': './outputs\\neus-dtu-Miku\\@202  
30723-131350\\save', 'ckpt_dir': './outputs\\neus-dtu-Miku\\@20230723-131350\\ckpt', 'code  
_dir': './outputs\\neus-dtu-Miku\\@20230723-131350\\code', 'config_dir': './outputs\\neus-  
dtu-Miku\\@20230723-131350\\config'}

```

## run.py

```
import argparse

def config_parser():
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args() 
    or 
    args, extras = parser.parse_known_args()
    
    return args
    or
    return args, extras
```

# process_data

```
{'pose': tensor([[[-0.0898,  0.8295, -0.5512,  1.3804],
         [ 0.0600,  0.5570,  0.8284, -2.0745],
         [ 0.9941,  0.0413, -0.0998,  0.0576],
         [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0'), 
'direction': tensor([[[[-0.3379, -0.5977,  1.0000],
          [-0.3354, -0.5977,  1.0000],
          [-0.3329, -0.5977,  1.0000],
          ...,
          [ 0.3341, -0.5977,  1.0000],
          [ 0.3366, -0.5977,  1.0000],
          [ 0.3391, -0.5977,  1.0000]],

         [[-0.3379, -0.5952,  1.0000],
          [-0.3354, -0.5952,  1.0000],
          [-0.3329, -0.5952,  1.0000],
          ...,
          [ 0.3341, -0.5952,  1.0000],
          [ 0.3366, -0.5952,  1.0000],
          [ 0.3391, -0.5952,  1.0000]],

         [[-0.3379, -0.5927,  1.0000],
          [-0.3354, -0.5927,  1.0000],
          [-0.3329, -0.5927,  1.0000],
          ...,
          [ 0.3341, -0.5927,  1.0000],
          [ 0.3366, -0.5927,  1.0000],
          [ 0.3391, -0.5927,  1.0000]],

         ...,

         [[-0.3379,  0.5940,  1.0000],
          [-0.3354,  0.5940,  1.0000],
          [-0.3329,  0.5940,  1.0000],
          ...,
          [ 0.3341,  0.5940,  1.0000],
          [ 0.3366,  0.5940,  1.0000],
          [ 0.3391,  0.5940,  1.0000]],

         [[-0.3379,  0.5965,  1.0000],
          [-0.3354,  0.5965,  1.0000],
          [-0.3329,  0.5965,  1.0000],
          ...,
          [ 0.3341,  0.5965,  1.0000],
          [ 0.3366,  0.5965,  1.0000],
          [ 0.3391,  0.5965,  1.0000]],

         [[-0.3379,  0.5990,  1.0000],
          [-0.3354,  0.5990,  1.0000],
          [-0.3329,  0.5990,  1.0000],
          ...,
          [ 0.3341,  0.5990,  1.0000],
          [ 0.3366,  0.5990,  1.0000],
          [ 0.3391,  0.5990,  1.0000]]]], device='cuda:0'), 
'index': tensor([124]), 'H': ['480'], 'W': ['272'], 
'image': tensor([[[[0.3242, 0.3438, 0.3164],
          [0.3281, 0.3477, 0.3203],
          [0.3320, 0.3398, 0.3125],
          ...,
          [0.0469, 0.0469, 0.0469],
          [0.0469, 0.0469, 0.0469],
          [0.0469, 0.0469, 0.0469]],

         [[0.2969, 0.3164, 0.2891],
          [0.3008, 0.3203, 0.2930],
          [0.3125, 0.3203, 0.2930],
          ...,
          [0.0469, 0.0469, 0.0469],
          [0.0469, 0.0469, 0.0469],
          [0.0469, 0.0469, 0.0469]],

         [[0.2812, 0.3008, 0.2734],
          [0.2578, 0.2773, 0.2500],
          [0.2812, 0.2891, 0.2617],
          ...,
          [0.0469, 0.0469, 0.0469],
          [0.0469, 0.0469, 0.0469],
          [0.0469, 0.0469, 0.0469]],

         ...,

         [[0.5977, 0.6133, 0.5938],
          [0.5977, 0.6133, 0.5938],
          [0.5977, 0.6133, 0.5938],
          ...,
          [0.6328, 0.7578, 0.8828],
          [0.6328, 0.7578, 0.8828],
          [0.6289, 0.7539, 0.8789]],

         [[0.5977, 0.6133, 0.5938],
          [0.5977, 0.6133, 0.5938],
          [0.5977, 0.6133, 0.5938],
          ...,
          [0.6328, 0.7578, 0.8828],
          [0.6328, 0.7578, 0.8828],
          [0.6328, 0.7578, 0.8828]],

         [[0.5977, 0.6133, 0.5938],
          [0.5977, 0.6133, 0.5938],
          [0.5977, 0.6133, 0.5938],
          ...,
          [0.6289, 0.7539, 0.8789],
          [0.6328, 0.7578, 0.8828],
          [0.6328, 0.7578, 0.8828]]]], device='cuda:0'), 
  'mask': tensor([[[0.9961, 0.9961, 0.9961,  ..., 0.9961, 0.9961, 0.9961],
         [0.9961, 0.9961, 0.9961,  ..., 0.9961, 0.9961, 0.9961],
         [0.9961, 0.9961, 0.9961,  ..., 0.9961, 0.9961, 0.9961],
         ...,
         [0.9961, 0.9961, 0.9961,  ..., 0.9961, 0.9961, 0.9961],
         [0.9961, 0.9961, 0.9961,  ..., 0.9961, 0.9961, 0.9961],
         [0.9961, 0.9961, 0.9961,  ..., 0.9961, 0.9961, 0.9961]]],
       device='cuda:0')}
```

# NeRF个人理解

## 20230705-NeRF_Neus_InstantNGP

基于NeRF的方法主要包括以下部分：

- 神经网络结构-->训练出来模型
- 位置编码方式：将点云位置使用编码得到高频的信息
- 体渲染函数：
  - 不透明度，累计透光率，权重，颜色
- 采样点的采样方式(精采样)
- 光线的生成方式，near和far的计算方式

在NeRF的基础上生成mesh模型：需要确定物体的表面，用不同的方法可以生成不同的隐式模型，如NeRF为位置转密度颜色，Neus为位置转SDF。以空间原点为中心，根据bound_min和bound_max生成一个resolution x resolution x resolution立方点云模型，根据隐式模型，生成其中每个点的密度颜色或者sdf值，然后选择零水平集为物体的表面，根据物体表面上的点生成三角形网格，并得到mesh模型。


## 质量评估指标

在标准设置中通过NeRF进行的新颖视图合成使用了视觉质量评估指标作为基准。这些指标试图评估单个图像的质量，要么有(完全参考)，要么没有(无参考)地面真相图像。峰值信噪比(PSNR)，结构相似指数度量(SSIM)[32]，学习感知图像补丁相似性(LPIPS)[33]是目前为止在NeRF文献中最常用的。

### PSNR↑
PSNR是一个无参考的质量评估指标
$PSNR(I)=10\cdot\log_{10}(\dfrac{MAX(I)^2}{MSE(I)})$

### SSIM↑
SSIM是一个完整的参考质量评估指标。
$SSIM(x,y)=\dfrac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}$

### LPIPS↓
LPIPS是一个完整的参考质量评估指标，它使用了学习的卷积特征。分数是由多层特征映射的加权像素级MSE给出的。
$LPIPS(x,y)=\sum\limits_{l}^{L}\dfrac{1}{H_lW_l}\sum\limits_{h,w}^{H_l,W_l}||w_l\odot(x^l_{hw}-y^l_{hw})||^2_2$

## Dataset


DTU. The DTU dataset [Large Scale Multi-view Stereopsis Evaluation-论文阅读讨论-ReadPaper](https://readpaper.com/paper/2085905957) consists of different static scenes with a wide variety of materials, appearance, and geometry, where each scene contains 49 or 64 images with the resolution of 1600 x 1200. We use the same 15 scenes as IDR [[PDF] SPIDR: SDF-based Neural Point Fields for Illumination and Deformation-论文阅读讨论-ReadPaper](https://readpaper.com/paper/4679926840484184065) to evaluate our approach. Experiments are conducted to investigate both the with (w/) and without (w/o) foreground mask settings. As DTU provides the ground truth point clouds, we measure the recovered surfaces through the commonly studied Chamfer Distance (CD) for quantitative comparisons.

BlendedMVS. The BlendedMVS dataset [[PDF] BlendedMVS: A Large-scale Dataset for Generalized Multi-view Stereo Networks-论文阅读讨论-ReadPaper](https://readpaper.com/paper/2990386223) consists of a variety of complex scenes, where each scene provides 31 to 143 multi-view images with the image size of 768 ×576. We use the same 7 scenes as NeuS [[PDF] NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction-论文阅读讨论-ReadPaper](https://readpaper.com/paper/3173522942) to validate our method. We only present qualitative comparisons on this dataset, because the ground truth point clouds are not available.


数据集的构建依赖Colmap即SFM和Multi-View Stereo
- Structure-from-Motion Revisited
    - Feature detection and extraction
    - Feature matching and geometric verification
    - Structure and motion reconstruction
- Pixelwise View Selection for Unstructured Multi-View Stereo
    - get a dense point cloud

> [colmap tutorial](https://colmap.github.io/tutorial.html)

[SFM算法原理初简介 | jiajie (gitee.io)](https://jiajiewu.gitee.io/post/tech/slam-sfm/sfm-intro/)

![image.png|500](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230717204531.png)

SFM - 稀疏重建: 
- 特征提取 , ref: [非常详细的sift算法原理解析_可时间倒数了的博客-CSDN博客](https://blog.csdn.net/u010440456/article/details/81483145)
    - eg：sift算法(Scale-invariant feature transform)是一种电脑视觉的算法用来侦测与描述影像中的局部性特征，它在空间尺度中寻找极值点，并提取出其位置、尺度、旋转不变量，此算法由 David Lowe在1999年所发表，2004年完善总结
- 特征匹配，ref: [sfm流程概述_神气爱哥的博客-CSDN博客](https://blog.csdn.net/qingcaichongchong/article/details/62424661)


## Neus

DTU、BlendedMVS、Neus_custom_data

Neus: [NeuS/preprocess_custom_data at main · Totoro97/NeuS (github.com)](https://github.com/Totoro97/NeuS/tree/main/preprocess_custom_data)

```
<case_name>
|-- cameras_xxx.npz    # camera parameters
    |-- camera_mat_{}.npy，intrinsic
    |-- camera_mat_inv_{}.npy
    |-- world_mat_{}.npy，intrinsic @ w2c --> world to pixel
    |-- world_mat_inv_{}.npy
    |-- scale_mat_{}.npy ，根据手动清除point得到的sparse_points_interest.ply
    |-- scale_mat_inv_{}.npy
|-- image
    |-- 000.png        # target image for each view
    |-- 001.png
    ...
|-- mask
    |-- 000.png        # target mask each view (For unmasked setting, set all pixels as 255)
    |-- 001.png
    ...
```

## NeRF

nerf_synthetic
nerf_llff_data

# BUG

```
Error
1.
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 26.00 MiB (GPU 0; 23.69 GiB total capacity; 20.89 GiB already allocated; 23.69 MiB free; 22.11 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF  0% 0/2 [00:01<?, ?it/s]

with torch.no_grad():
    val_epoch

2.
[2023-07-25 21:10:06,584] INFO: ==>Training Epoch 1, lr = 0.010000  
loss=0.3620 (nan), lr=0.008147: : 100% 178/178 [00:15<00:00, 11.85it/s]  
[2023-07-25 21:10:21,609] INFO: ==>Training Epoch 2, lr = 0.008147  
loss=0.1856 (0.3541), lr=0.006637: : 100% 178/178 [00:13<00:00, 13.39it/s]  
[2023-07-25 21:10:34,907] INFO: ==>Training Epoch 3, lr = 0.006637  
loss=0.1963 (0.2716), lr=0.005408: : 100% 178/178 [00:13<00:00, 13.45it/s]  
[2023-07-25 21:10:48,139] INFO: ==>Training Epoch 4, lr = 0.005408  
loss=nan (nan), lr=0.004406: : 100% 178/178 [00:06<00:00, 26.05it/s]  
[2023-07-25 21:10:54,974] INFO: ==>Training Epoch 5, lr = 0.004406  
loss=nan (nan), lr=0.003589: : 100% 178/178 [00:03<00:00, 46.24it/s]  
[2023-07-25 21:10:58,824] INFO: ==>Validation at epoch 5  
0% 0/2 [00:00<?, ?it/s]/root/NeRF-Mine/utils/mixins.py:160: RuntimeWarning: invalid value encountered i  
n divide  
img = (img - img.min()) / (img.max() - img.min())  
/root/NeRF-Mine/utils/mixins.py:169: RuntimeWarning: invalid value encountered in cast  
img = (img * 255.).astype(np.uint8)  
psnr=4.844281196594238: : 100% 2/2 [00:05<00:00, 2.52s/it]  
[2023-07-25 21:11:03,865] INFO: ==>Training Epoch 6, lr = 0.003589  
loss=nan (nan), lr=0.002924: : 100% 178/178 [00:03<00:00, 47.74it/s]  
[2023-07-25 21:11:07,595] INFO: ==>Training Epoch 7, lr = 0.002924

inv_s  ----> nan ， loss_mask ----> nan , loss_eikonal ----> nan
都有问题
sdf_grad_samples: 突然变为 0,3

guess1: lr太高1e-2 √

3.
训练出来的test_Video和mesh位置不准确 --> 数据集加载时的c2w没处理

4.
训练的效果不好且很慢 --> 没有使用processHashGrid
lr太低可能陷入局部最优

NOTE：训练过程中loss突然变得很大
TODO：防止过拟合-->添加 torch.cuda.amp.GradScaler() 解决 loss为nan或inf的问题
```