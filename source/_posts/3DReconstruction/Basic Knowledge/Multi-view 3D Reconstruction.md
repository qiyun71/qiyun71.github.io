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

三维重建是计算机视觉和计算机图像图形学相结合的一个热门研究方向。根据测量时是否与被测物体接触，可分为接触式测量和非接触式测量。
- 接触式测量方法虽然测量精度高，但测量效率低，速度慢，操作不当很容易损坏被测物体表面，而且由于探头有一定表面积，对表面复杂的物体难以测量，不具备普遍性和通用性。
- 非接触式三维测量方式又可以分为两大类：主动式测量和被动式测量。非接触式测量方式以其无损坏、测量速度高、简单等优点已成为三维轮廓测量的研究趋势。
  - 主动式测量是向目标物体表面投射设计好的图案，该图案由于物体的高度起伏引起一定的畸变，通过匹配畸变的图案获得目标物体的。**TOF、结构光三维重建**
    - 结构光三维重建 [Stereo Matching Review](/3DReconstruction/Other%20Methods/Stereo%20Matching/Stereo%20Matching%20Review)
  - 被动式测量是通过周围环境光对目标物体进行照射，然后检测目标物体的特征点以得到其数据。**双目/多目视觉法、SFM、MVS、NeRF**
    - 双目立体匹配 [Structured Light Review](/3DReconstruction/Other%20Methods/Structured%20Light/Structured%20Light%20Review)
    - 单目RGB相机 
    - 单目RGBD相机
      - Neural RGB-D Surface Reconstruction
      - BID-NeRF: RGB-D image pose estimation with inverted Neural Radiance Fields

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231219125844.png)

被动式单目视觉的多视图三维重建，目前重建结果最好的是基于 **NeRF** 的多视图重建方法。NeRF基本流程为从相机位姿出发，得到多条从相机原点到图片像素的光线，在光线上进行采样得到一系列空间点，然后对采样点坐标进行编码，输入密度MLP网络进行计算，得到采样点位置的密度值，同时对该点的方向进行编码，输入颜色MLP网络计算得到该点的颜色值。然后根据体渲染函数沿着光线积分，得到像素预测的颜色值并与真实的颜色值作损失，优化MLP网络参数，最后得到一个用MLP参数隐式表达的三维模型。为了从隐式函数中提取显示模型，需要使用MarchingCube得到物体表面的点云和网格。

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231219125859.png)

应用：
- [快手智能3D物体重建系统解析 (qq.com)](https://mp.weixin.qq.com/s/-VU-OBpdmU0DLiEgtTFEeg)
- [三维重建如今有什么很现实的应用吗？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/449185693)

---

***基于NeRF的多视图三维重建***

# Abstract

# Introduction+RelatedWork

COLMAP：
- SFM(Structure from motion)，估计相机位姿，特征点的稀疏点云
- MVS(Multi-View Stereo)，估计深度图，深度图融合稠密点云
- 泊松表面重建(Screened Poisson Surface Reconstruction)，稠密点云重建网格

## 传统的多视图三维重建方法

- 基于点云
  - SFM
- 基于网格
- 基于体素
- 基于深度图
  - [MVSNet: Depth Inference for Unstructured Multi-view Stereo (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4518062699161739265&noteId=1986540055632613120)
  - [MVS: Multi-View Stereo based on deep learning. | Learning notes, codes and more. (github.com)](https://github.com/doubleZ0108/MVS)
  - [XYZ-qiyh/multi-view-3d-reconstruction: 📷 基于多视角图像的三维重建 (github.com)](https://github.com/XYZ-qiyh/multi-view-3d-reconstruction)

## 基于NeRF的重建方法

### 基于隐式表示的重建方法
- [occupancy_networks: This repository contains the code for the paper "Occupancy Networks - Learning 3D Reconstruction in Function Space" (github.com)](https://github.com/autonomousvision/occupancy_networks)
- [facebookresearch/DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation (github.com)](https://github.com/facebookresearch/DeepSDF?tab=readme-ov-file)

**NeRF被提出**
![Network.png|666](https://raw.githubusercontent.com/yq010105/Blog_images/main/Network.png)

### 基于神经辐射场的重建方法

- 快：Plenoxels、**InstantNGP**
- 好：[UNISURF](https://github.com/autonomousvision/unisurf)、VolSDF、**NeuS**
- InstantNGP+NeuS：Neuralangelo、PermutoSDF、NeuS2、NeuDA、Instant-NSR、BakedSDF

# Method

## 数据采集平台搭建

研究角度：数据集RGB+Depth验证方法可行性与优越性
- 数据集制作需要RGBD相机+三维扫描仪
工程角度：多角度RGBD相机重建高精度模型


Color-NeuS: 
- 三维扫描仪[EinScan Pro 2X - Shining3D Multifunctional Handheld Scanner | EinScan](https://www.einscan.com/handheld-3d-scanner/einscan-pro-2x-2020/)