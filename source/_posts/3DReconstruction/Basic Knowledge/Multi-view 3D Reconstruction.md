---
title: Multi-view 3D Reconstruction
date: 2023-10-23 19:54:54
tags:
  - 3DReconstruction
categories: 3DReconstruction/Basic Knowledge
top: true
---

| 3D Reconstruction |             Single-view              |             Multi-view             |
|:-----------------:|:------------------------------------:|:----------------------------------:|
|       特点        | **简单但信息不足，未见区域很难重建** | **多视图信息互补但一致性很难保证** |
|  深度估计 **[DE](/3DReconstruction/Basic%20Knowledge/Other%20Paper%20About%20Reconstruction)**  |              2K2K,ECON               |          MVS,MVSNet-based          |
|  隐式函数 **[IF](/3DReconstruction/Basic%20Knowledge/Other%20Paper%20About%20Reconstruction)**  |              PIFu,ICON               |    NeuS,DoubleField,SuGaR    |
|  生成模型 **[GM](/3DReconstruction/Basic%20Knowledge/Generative%20Models%20Review)**  |           BuilDIff, SG-GAN           |            DiffuStereo             |
|  混合方法 **HM**  |                 HaP                  |               DMV3D                |

NeRF：[NeRF-review](/3DReconstruction/Basic%20Knowledge/NeRF/NeRF-review) | [NeRF-Mine](/3DReconstruction/Basic%20Knowledge/NeRF/NeRF-Mine)

Follow: [NeRF and Beyond日报](https://www.zhihu.com/column/c_1710703836652716032) | [nerf and beyond docs](https://github.com/yangjiheng/nerf_and_beyond_docs) | **[ventusff/neurecon](https://github.com/ventusff/neurecon)** | [Surface Reconstruction](https://paperswithcode.com/task/surface-reconstruction) | [传统3D Reconstruction](https://github.com/openMVG/awesome_3DReconstruction_list)
Blog: [Jianfei Guo](https://longtimenohack.com/) | 
人体: [Multi-view Human Body Reconstruction](/3DReconstruction/Basic%20Knowledge/Multi-view%20Human%20Body%20Reconstruction)

<!-- more -->

# Review

## Following



## 重建方法分类
三维重建是计算机视觉和计算机图像图形学相结合的一个热门研究方向。根据测量时是否与被测物体接触，可分为接触式测量和非接触式测量。
- 接触式测量方法虽然测量精度高，但测量效率低，速度慢，操作不当很容易损坏被测物体表面，而且由于探头有一定表面积，对表面复杂的物体难以测量，不具备普遍性和通用性。
- 非接触式三维测量方式又可以分为两大类：主动式测量和被动式测量。非接触式测量方式以其无损坏、测量速度高、简单等优点已成为三维轮廓测量的研究趋势。
  - 主动式测量是向目标物体表面投射设计好的图案，该图案由于物体的高度起伏引起一定的畸变，通过匹配畸变的图案获得目标物体的。**TOF、结构光三维重建**
    - 结构光三维重建 [Structured Light Review](/3DReconstruction/Other%20Methods/Structured%20Light/Structured%20Light%20Review)
  - 被动式测量是通过周围环境光对目标物体进行照射，然后检测目标物体的特征点以得到其数据。**双目/多目视觉法、SFM、MVS、NeRF**
    - 双目立体匹配 [Stereo Matching Review](/3DReconstruction/Other%20Methods/Stereo%20Matching/Stereo%20Matching%20Review)
    - NeRF+RGBD
      - Neural RGB-D Surface Reconstruction
      - BID-NeRF: RGB-D image pose estimation with inverted Neural Radiance Fields

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231219125844.png)

## NeRF基本流程
被动式单目视觉的多视图三维重建，目前重建结果最好的是基于 **NeRF** 的多视图重建方法。NeRF基本流程为从相机位姿出发，得到多条从相机原点到图片像素的光线，在光线上进行采样得到一系列空间点，然后对采样点坐标进行编码，输入密度MLP网络进行计算，得到采样点位置的密度值，同时对该点的方向进行编码，输入颜色MLP网络计算得到该点的颜色值。然后根据体渲染函数沿着光线积分，得到像素预测的颜色值并与真实的颜色值作损失，优化MLP网络参数，最后得到一个用MLP参数隐式表达的三维模型。为了从隐式函数中提取显示模型，需要使用MarchingCube得到物体表面的点云和网格。

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231219125859.png)

应用：
- [快手智能3D物体重建系统解析 (qq.com)](https://mp.weixin.qq.com/s/-VU-OBpdmU0DLiEgtTFEeg)
- [三维重建如今有什么很现实的应用吗？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/449185693)

拓展阅读
- [“三维AIGC与视觉大模型”十五问 (qq.com)](https://mp.weixin.qq.com/s?__biz=MzI0MTY1NTk1Nw==&mid=2247495573&idx=1&sn=968b2d4fe20e1ab21e139f943b3cce71&chksm=e90ae66fde7d6f79cc842d9cde6b928605e3d360d17e1fdf9bde7c854058f1649a1bc45e53a7&scene=132&exptype=timeline_recommend_article_extendread_samebiz#wechat_redirect)

---

***基于NeRF的多视图三维重建***

研究任务与目的：设计一套快速高精度的低成本无接触三维重建系统，用以快速地在日常生活领域生成三维模型，然后进行3D打印，满足用户定制化模型的需求

# Abstract

# Introduction+RelatedWork

COLMAP：
- SFM(Structure from motion)，估计相机位姿，特征点的稀疏点云
- MVS(Multi-View Stereo)，估计深度图，深度图融合稠密点云
- 泊松表面重建(Screened Poisson Surface Reconstruction)，稠密点云重建网格

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

### 基于隐式表示的重建方法
- [occupancy_networks: This repository contains the code for the paper "Occupancy Networks - Learning 3D Reconstruction in Function Space" (github.com)](https://github.com/autonomousvision/occupancy_networks)
- [facebookresearch/DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation (github.com)](https://github.com/facebookresearch/DeepSDF?tab=readme-ov-file)

Occupancy Networks 与 DeepSDF 依然需要显示的三维模型作监督

**NeRF被提出(2020 by UC Berkeley)**[NeRF: Neural Radiance Fields (matthewtancik.com)](https://www.matthewtancik.com/nerf)
![Network.png|666](https://raw.githubusercontent.com/yq010105/Blog_images/main/Network.png)

- 优点：隐式表示低内存、自监督方法(成本低)、单个场景进行训练可以重建任意物体=(优点or缺点)=泛化性差
- 缺点：重建速度慢、重建精度差、所需图像数量多、适用场景单一(限于有界场景、远处模糊，出现伪影)

### 基于神经辐射场重建方法的改进

重建速度+重建精度
- 更快：Plenoxels、**InstantNGP**
- 更好：[UNISURF](https://github.com/autonomousvision/unisurf)、VolSDF、**NeuS**
- 快+好(InstantNGP+NeuS)：Neuralangelo、PermutoSDF、NeuS2、NeuDA、Instant-NSR、BakedSDF

重建所需图像数量
- SparseNeuS、NeuSurf、FORGE、FreeNeRF、ZeroRF、ColNeRF、SparseNeRF、pixelNeRF

无界场景？

远近细节比例不平衡
- Mip-NeRF、Mip-NeRF 360、Zip-NeRF

# Method

## 数据采集平台搭建

研究角度：数据集RGB+Depth验证方法可行性与优越性
- 数据集制作需要RGBD相机+三维扫描仪
工程角度：多角度RGBD相机重建高精度模型

Color-NeuS: 
- 三维扫描仪[EinScan Pro 2X - Shining3D Multifunctional Handheld Scanner | EinScan](https://www.einscan.com/handheld-3d-scanner/einscan-pro-2x-2020/)

## 2024-0101 ~ 0113

重建项目的具体想法：
- 结构光相机的选型
- 扫描仪选型

# 实验

| 实验时间         | 对象        | 方法                               |
| ---------------- | ----------- | ---------------------------------- |
| @20240108-124117 | dtu114_mine | neus                               |
| @20240108-133914 | dtu114_mine | + ProgressiveBandHashGrid          |
| @20240108-151934 | dtu114_mine | + loss_curvature(sdf_grad_samples) |
|                  |             |                                    |

