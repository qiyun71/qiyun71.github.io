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

Follow: [NeRF and Beyond日报](https://www.zhihu.com/column/c_1710703836652716032) | [nerf and beyond docs](https://github.com/yangjiheng/nerf_and_beyond_docs) | **[ventusff/neurecon](https://github.com/ventusff/neurecon)** | [Surface Reconstruction](https://paperswithcode.com/task/surface-reconstruction)
Blog: [Jianfei Guo](https://longtimenohack.com/) | 
人体: [Multi-view Human Body Reconstruction](/3DReconstruction/Basic%20Knowledge/Multi-view%20Human%20Body%20Reconstruction)

<!-- more -->

三维重建方向目前重建结果最好的是基于 **NeRF** 的多视图重建方法，研究主题为从多视图姿势图片中重建出物体的表面几何和外观，基本流程为从相机位姿开始，得到多条从相机原点到图片像素的光线，在光线上进行采样得到一系列空间点，然后对采样点坐标进行编码，输入密度MLP网络进行计算，得到采样点位置的密度值，同时对该点的方向进行编码，输入颜色MLP网络计算得到该点的颜色值。然后根据体渲染函数沿着光线积分，得到像素预测的颜色值并与真实的颜色值作损失，优化MLP网络参数，最后得到一个用MLP参数隐式表达的三维模型。

**Neural 3D Reconstruction** = Volume rendering + 3D implicit surface(eg: SDF in NeuS)

framework overview image from [ventusff/neurecon](https://github.com/ventusff/neurecon)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231130152644.png)

---

# Abstract

# Introduction

三维重建是指从单幅或多幅二维图像中重建出物体的三维模型并对三维模型进行纹理映射的过程
>[深度学习背景下的图像三维重建技术进展综述 (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=2073861861536518400&noteId=2073862022044072960)

目前，三维重建技术已在游戏、电影、测绘、定位、导航、自动驾驶、VR/AR、工业制造以及消费品领域等方面得到了广泛的应用。
> [三维重建算法综述|传统+深度学习方式 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/108422696)

三维模型本质上是计算机内的一组数据，在建模初期通常依赖人工运用专业软件进行创建，然而，这种传统方法存在着高昂的时间成本和对专业技能的需求。随着科技的进步，特别是计算机视觉和传感器技术的发展，激光扫描仪等设备的应用使得对实际物体的几何结构进行数字化变得更加容易，然而由于激光扫描仪价格昂贵，其广泛应用受到一定限制。当前，计算机技术的飞速发展为三维重建领域注入了新的活力，多视图方法通过融合多个角度的图像信息，提高了对目标物体几何形状的还原能力，且图像三维重建成本低、操作简单，可以对不规则的自然或人工合成物体进行建模，重建真实物体的三维模型

多视图立体(MVS)是指利用多张 RGB 图像，来恢复场景或者物体的轮廓。从具有一定重叠度的多视图视角中恢复场景的稠密结构的技术，传统方法利用几何、光学一致性构造匹配代价，进行匹配代价累积，再估计深度值。虽然传统方法有较高的深度估计精度，但由于存在在缺少纹理或者光照条件剧烈变化的场景中的错误匹配
> [MVSNet: Depth Inference for Unstructured Multi-view Stereo (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4518062699161739265&noteId=1986540055632613120)

基于神经隐式表征的三维重建最近成为经典重建方法的一种非常有前途的替代方法[37,8,2]，因为它具有高重建质量，并且具有重建经典方法难以重建的复杂物体的潜力，例如非朗伯曲面和薄结构。最近的作品将曲面表示为符号距离函数(SDF)[49,52,17,23]或占用[30,31]。为了训练神经模型，这些方法使用可微表面渲染方法将3D物体渲染成图像，并将其与输入图像进行比较以进行监督。
如IDR产生了令人印象深刻的重建结果，但它无法重建结构复杂且导致深度突变的物体。造成这种限制的原因是IDR中使用的表面渲染方法只考虑每个光线的单个表面交叉点。因此，梯度只存在于这个单点，对于有效的反向传播来说，这个点太局部了，当图像上的深度发生突变时，优化会陷入一个较差的局部最小值。
> [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=4718711588576575489&noteId=1791151226962648064)

NeuS

加速：Neuralangelo、PermutoSDF、NeuS2

三维重建任务根据输入视图的数量可以划分为单视图重建与多视图重建。单视图重建使用单个视图作为输入，天然的会缺乏物体未见部分的信息，虽然可以使用大规模的数据集来让网络学习缺失区域的形状，但想要得到高保真的重建是非常困难的。而多视图重建
此外，一般的单视图生成式方法需要 3D 模型数据集作监督，这些模型数据需要花费大量的人工成本和时间成本，一般很难得到。

Multi-view 3D Reconstruction 按照物体规模可分：
- **Indoor** Scene Reconstruction： NeuRIS 室内
  - [Human Body](Multi-view%20Human%20Body%20Reconstruction.md)：DiffuStereo, DoubleField
  - Object：NeuS, Neuralangelo, Adaptive Shells
- **Outdoor** Scene Reconstruction
  - Unbounded, Urban：Mip-NeRF 360, UrbanRF
  - Street (Autonomous Driving)：StreetSurf

基于 NeRF 进行三维重建的基本流程：
图像拍摄 + (--> SFM 估计相机位姿) --> 训练 NeRF 网络/隐式函数 --> Marching Cube 提取 mesh
- 估计相机位姿: COLMAP *位姿有误差* 影响重建精度
- 训练 NeRF 网络(自监督): Density-Field, SDF-Field, Color-Field
- MC 提取 mesh: Trimesh, PyMCubes, CuMCubes

# Related Work




