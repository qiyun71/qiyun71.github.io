---
title: Multi-view 3D Object Reconstruction
date: 2023-10-23 19:54:54
tags:
  - 3DReconstruction
categories: 3DReconstruction/Basic Knowledge
top: true
---

|                                       3D Reconstruction                                       |             Single-view              |             Multi-view             |
|:---------------------------------------------------------------------------------------------:|:------------------------------------:|:----------------------------------:|
|                                             特点                                              | **简单但信息不足，未见区域很难重建** | **多视图信息互补但一致性很难保证** |
| 深度估计 **[DE](/3DReconstruction/Basic%20Knowledge/Other%20Paper%20About%20Reconstruction)** |              2K2K,ECON               |          MVS,MVSNet-based          |
| 隐式函数 **[IF](/3DReconstruction/Basic%20Knowledge/Other%20Paper%20About%20Reconstruction)** |              PIFu,ICON               |       NeuS,DoubleField,SuGaR       |
|      生成模型 **[GM](Generative%20Models%20Reconstruction.md)**      |           BuilDIff, SG-GAN           |            DiffuStereo             |
|                                        混合方法 **HM**                                        |                 HaP                  |               DMV3D                |
| 显式表示 ER                                                                                              | Pixel2Mesh++                                     | Pixel2Mesh                                   |

NeRF：[NeRF-review](/3DReconstruction/Basic%20Knowledge/NeRF/NeRF-review) | [NeRF-Mine](/3DReconstruction/Basic%20Knowledge/NeRF/NeRF-Mine)

Follow: [NeRF and Beyond日报](https://www.zhihu.com/column/c_1710703836652716032) | [nerf and beyond docs](https://github.com/yangjiheng/nerf_and_beyond_docs) | **[ventusff/neurecon](https://github.com/ventusff/neurecon)** | [Surface Reconstruction](https://paperswithcode.com/task/surface-reconstruction) | [传统3D Reconstruction](https://github.com/openMVG/awesome_3DReconstruction_list)
Blog: [Jianfei Guo](https://longtimenohack.com/) | 
人体: [Multi-view Human Body Reconstruction](/3DReconstruction/Basic%20Knowledge/Multi-view%20Human%20Body%20Reconstruction)
评价指标/Loss: [Metrics](/3DReconstruction/Basic%20Knowledge/NeRF/NeRF/Metrics)
数据集: [Datasets](/3DReconstruction/Basic%20Knowledge/NeRF/NeRF/Datasets)

<!-- more -->

# Review

## 三维重建分类

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231219125844.png)

## NeRF基本流程

多视图三维重建，目前较好的方法是在NeuS和HashGrid方法基础上的改进。
NeRF基本流程为从相机位姿出发，得到多条从相机原点到图片像素的光线(**像素选取方法**)，在光线上进行采样得到一系列空间点(**采样方式**)，然后对采样点坐标进行编码(**编码方式**)，输入密度MLP网络进行计算(**神经网络结构**)，得到采样点位置的密度值，同时对该点的方向进行编码，输入颜色MLP网络计算得到该点的颜色值。然后根据体渲染函数沿着光线积分(**体渲染函数**)，得到像素预测的颜色值并与真实的颜色值作损失(**损失函数**)，优化MLP网络参数，最后得到一个用MLP参数隐式表达的三维模型。为了从隐式函数中提取显示模型，需要使用**MarchingCube**得到物体表面的点云和网格。

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231219125859.png)

应用：
- [快手智能3D物体重建系统解析 (qq.com)](https://mp.weixin.qq.com/s/-VU-OBpdmU0DLiEgtTFEeg)
- [三维重建如今有什么很现实的应用吗？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/449185693)
- LumaAI

拓展阅读
- [“三维AIGC与视觉大模型”十五问 (qq.com)](https://mp.weixin.qq.com/s?__biz=MzI0MTY1NTk1Nw==&mid=2247495573&idx=1&sn=968b2d4fe20e1ab21e139f943b3cce71&chksm=e90ae66fde7d6f79cc842d9cde6b928605e3d360d17e1fdf9bde7c854058f1649a1bc45e53a7&scene=132&exptype=timeline_recommend_article_extendread_samebiz#wechat_redirect)

---

研究任务与目的：设计一套快速高精度的低成本无接触三维重建系统，用以快速地在日常生活领域生成三维模型，然后进行3D打印，满足用户定制化模型的需求

# Abstract

# Introduction+RelatedWork

## 传统的多视图三维重建方法

- 基于点云PointCloud **SFM**
- 基于网格Surface Grid
- 基于体素Voxel
- 基于深度图Depth **MVS**
  - [MVSNet: Depth Inference for Unstructured Multi-view Stereo (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4518062699161739265&noteId=1986540055632613120)
  - [MVS: Multi-View Stereo based on deep learning. | Learning notes, codes and more. (github.com)](https://github.com/doubleZ0108/MVS)
  - [XYZ-qiyh/multi-view-3d-reconstruction: 📷 基于多视角图像的三维重建 (github.com)](https://github.com/XYZ-qiyh/multi-view-3d-reconstruction)

对场景显式的表征形式：
- 优点是能够对场景进行显示建模从而合成照片级的虚拟视角
- 缺点是这种离散表示因为不够精细化会造成重叠等伪影，而且最重要的，它们对内存的消耗限制了高分辨率场景的应用

## 基于NeRF的重建方法

### 基于隐式表示的三维重建方法
- [occupancy_networks: This repository contains the code for the paper "Occupancy Networks - Learning 3D Reconstruction in Function Space" (github.com)](https://github.com/autonomousvision/occupancy_networks)
- [facebookresearch/DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation (github.com)](https://github.com/facebookresearch/DeepSDF?tab=readme-ov-file)

Occupancy Networks 与 DeepSDF 依然需要显示的三维模型作监督

### 基于神经辐射场重建的三维重建方法

**NeRF被提出(2020 by UC Berkeley)**[NeRF: Neural Radiance Fields (matthewtancik.com)](https://www.matthewtancik.com/nerf)
![Network.png|666](https://raw.githubusercontent.com/yq010105/Blog_images/main/Network.png)

- 优点：隐式表示低内存、自监督方法(成本低)、单个场景进行训练可以重建任意物体=(优点or缺点)=泛化性差
- 缺点：重建速度慢、重建精度差、所需图像数量多、适用场景单一(限于有界场景、远处模糊，出现伪影)

### NeRF的不足

重建速度+重建精度
- 更快：Plenoxels、**InstantNGP**
- 更好：[UNISURF](https://github.com/autonomousvision/unisurf)、VolSDF、**NeuS**
- 快+好(InstantNGP+NeuS)：Neuralangelo、PermutoSDF、NeuS2、NeuDA、Instant-NSR、BakedSDF

重建所需图像数量
- SparseNeuS、NeuSurf、FORGE、FreeNeRF、ZeroRF、ColNeRF、SparseNeRF、pixelNeRF

远近细节比例不平衡（物体不在相机景深导致的模糊）
- Mip-NeRF、Mip-NeRF 360、Zip-NeRF

相机参数有误差

照片质量不好（高光、阴影、HDR|LDR）

### 目前方法的不足

重建质量能否更好，重建速度能否更快

# Method

## 大论文章节

- 基于神经隐式表面和体渲染的三维重建 NeuS
  - 采样方式
  - 编码方式
  - 神经网络结构
  - 体渲染函数
  - 损失函数

两种结构：
- 分步骤的(但是步骤之间关联要很小)
- 分方法改进的

## 数据采集平台搭建

Color-NeuS: 
- 三维扫描仪[EinScan Pro 2X - Shining3D Multifunctional Handheld Scanner | EinScan](https://www.einscan.com/handheld-3d-scanner/einscan-pro-2x-2020/)

DTU数据集：[Large Scale Multi-view Stereopsis Evaluation (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=732026274103447552&noteId=2151039906290343424)
- binary stripe encoding投影仪+相机（自制结构光相机） [DTU Robot Image Data Sets](http://roboimagedata.compute.dtu.dk/)
- 测量点云扫描精度：保龄球

BlendedMVS数据集：[YoYo000/BlendedMVS](https://github.com/YoYo000/BlendedMVS)
- Altizure.com online platform根据图片进行网格重建和位姿获取
- 根据带纹理网格生成图像数据集，目前无法评估点云[Point cloud evaluation · Issue #4](https://github.com/YoYo000/BlendedMVS/issues/4)

Tanks and Temples：[Tanks and Temples Benchmark](https://www.tanksandtemples.org/)
- 工业激光扫描仪(FARO Focus 3D X330 HDR scanner [Laser scanner](https://www.archiexpo.com/prod/faro/product-66338-1791336.html))捕获的模型
- 评估precision、recall、F-score(重建模型与GT模型)

# 实验

| 实验时间                                   |    对象     | 方法                               | 重建时间 |
|:------------------------------------------ |:-----------:| ---------------------------------- | -------- |
| @20240108-124117                           | dtu114_mine | neus + HashGrid                    |          |
| @20240108-133914                           | dtu114_mine | + ProgressiveBandHashGrid          |          |
| @20240108-151934                           | dtu114_mine | + loss_curvature(sdf_grad_samples) |          |
|                                            |             |                                    |          |
|                                            |  Miku_宿舍  | neus + HashGrid                    |          |
| @20240117-164156                           |  Miku_宿舍  | + ProgressiveBandHashGrid          | 47min    |
|                                            |             |                                    |          |
| @20240124-165842	 | TAT_Truck            | ProgressiveBandHashGrid                                   | 2h         |
| @20240124-230245                                           | TAT_Truck            | ProgressiveBandHashGrid                                   |          |
| @20240125-113410                                           | TAT_Truck            | ProgressiveBandHashGrid                                   |          |
