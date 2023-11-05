---
title: NeRF目前进展
date: 2023-06-25T13:01:00.000Z
tags:
  - NeRF
  - Review
categories: NeRF
date updated: 2023-08-11T17:30:59.000Z
---

NeRF相关的论文 at CVPR/ICCV/ECCV/NIPS/ICML/ICLR/SIGGRAPH

<!-- more -->

# NeRF

ECCV 2020 Oral - Best Paper Honorable Mention

| Year                                     |                                              Title&Project Page                                             | Brief Description | Conf/Jour |
| ---------------------------------------- | :---------------------------------------------------------------------------------------------------------: | :---------------: | :-------: |
| [2020](/NeRF/NeRF-Principle) | [NeRF:Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf) |        初始文        |    ECCV   |


## Volume Rendering Function

### PL-NeRF
[mikacuy/PL-NeRF: NeRF Revisited: Fixing Quadrature Instability in Volume Rendering, Neurips 2023 (github.com)](https://github.com/mikacuy/PL-NeRF)

NeRF 分段常数积分 --> PL-NeRF 分段线性积分
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231101151141.png)

## Efficiency

### Encoding

| Year                                                    |                                                                 Title&Project Page                                                                  |                                     Brief Description                                     |                               Conf/Jour                               |
| ------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|:---------------------------------------------------------------------:|
| [2022](/NeRF/Efficiency/NeRF-InstantNGP)                |                  [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://nvlabs.github.io/instant-ngp/)                   |                                     多分辨率哈希编码                                      |                ACM Transactions on Graphics (SIGGRAPH)                |
| [2023](/NeRF/SurfaceReconstruction/Neus-Instant-nsr-pl) |                                        [bennyguo/instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl)                                        |                                  Neus+NeRF+Nerfacc+tcnn                                   |                                 None                                  |
| [2023](/NeRF/SurfaceReconstruction/Neuralangelo)        |                   [Neuralangelo: High-Fidelity Neural Surface Reconstruction](https://research.nvidia.com/labs/dir/neuralangelo/)                   |                                 NGP_but数值梯度+Neus_SDF                                  | IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**) |
| [2023](/NeRF/SurfaceReconstruction/PermutoSDF)          |                                             [PermutoSDF](https://radualexandru.github.io/permuto_sdf/)                                              | NGP_butPermutohedral lattice + Neus_SDF，曲率损失和颜色MLP正则解决镜面+无纹理区域，更光滑 |                      IEEE/CVF Conference on CVPR                      |
| [2023](/NeRF/SurfaceReconstruction/NeuDA)               |                                                  [NeuDA](https://3d-front-future.github.io/neuda/)                                                  |                           NGP_butDeformable Anchors+HPE + Neus                            |                                 CVPR                                  |
| [2023](/NeRF/Efficiency/Zip-NeRF)                       |                                            [Zip-NeRF (jonbarron.info)](https://jonbarron.info/zipnerf/)                                             |                                     NGP + Mip-NeRF360                                     |                                 ICCV                                  |
| [2023](/NeRF/Efficiency/Tri-MipRF)                                    | [Tri-MipRF: Tri-Mip Representation for Efficient Anti-Aliasing Neural Radiance Fields (wbhu.github.io)](https://wbhu.github.io/projects/Tri-MipRF/) |                Tri-MipRF encoding(TensoRF + NGP)+ Cone Sampling(Mip-NeRF)                 |                                 ICCV                                  |
| [2022](/NeRF/Efficiency/TensoRF)                        |                          [TensoRF: Tensorial Radiance Fields (apchenstu.github.io)](https://apchenstu.github.io/TensoRF/)                           |                 TensoRF引入VM分解，提高重建的质量和速度，并减小了内存占用                 |                                 ECCV                                  |
| [2023](/NeRF/Efficiency/Strivec)                        |                                   [Zerg-Overmind/Strivec (github.com)](https://github.com/Zerg-Overmind/Strivec)                                    |                    局部张量CP分解为三向量+多尺度网格+占用网格采样方式                     |                                 ICCV                                  |
| [2023](/NeRF/PointCloud/3D%20Gaussian%20Splatting.md)   |        [3D Gaussian Splatting for Real-Time Radiance Field Rendering (inria.fr)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)         |            优化3D高斯点云，并使用splatting渲染方式，实现实时高分辨率训练+渲染             |                               SIGGRAPH                                |
| [2023](/NeRF/Efficiency/BakedSDF)                       |                                                       [BakedSDF](https://bakedsdf.github.io/)                                                       |                   VolSDF+MarchingCube得到mesh，漫反射+高斯叶来渲染颜色                    |                               SIGGRAPH                                |
|                                                         |                                                                                                                                                     |                                                                                           |                                                                       |

## Large Scale Scene

| Year                                                     | Title&Project Page                                                                                            | Brief Description                      | Conf/Jour |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------------- | :-------: |
| [2020](/NeRF/LargeScaleScene/NeRF++)         | [nerfplusplus: improves over nerf in 360 capture of unbounded scenes](https://github.com/Kai-46/nerfplusplus) | 将背景的采样点表示为四维向量，与前景分别使用不同的MLP进行训练       |   arXiv   |
| [2022](/NeRF/LargeScaleScene/Mip-NeRF%20360) | [mip-NeRF 360](https://jonbarron.info/mipnerf360/)                                                            | 将单位球外的背景参数化，小型提议网络进行精采样，正则化dist消除小区域云雾 |    CVPR   |

## PointCloud

| Year                                                  | Title&Project Page                                                                                                                                    | Brief Description                                                  |               Conf/Jour                |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |:--------------------------------------:|
| [2023](/NeRF/PointCloud/3D%20Gaussian%20Splatting.md) | [3D Gaussian Splatting for Real-Time Radiance Field Rendering (inria.fr)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)                  | 优化3D高斯点云，并使用splatting渲染方式，实现实时高分辨率训练+渲染 |                SIGGRAPH                |
| 2023                                                  | [Globally Consistent Normal Orientation for Point Clouds by Regularizing the Winding-Number Field](https://xrvitd.github.io/Projects/GCNO/index.html) | 使得稀疏点云和薄壁点云法向量一致                                   | ACM Transactions on Graphics(SIGGRAPH) |
| 2023                                                  | [Neural Kernel Surface Reconstruction](https://research.nvidia.com/labs/toronto-ai/NKSR/)                                                             | 从点云中进行表面重建                                               |          CVPR 2023 Highlight           |
| [2022](/NeRF/PointCloud/Point-NeRF)                   | [Point-NeRF: Point-based Neural Radiance Fields](https://xharlie.github.io/projects/project_sites/pointnerf/)                                         | 生成初始化点云，基于点云进行增删和体渲染                                                                   |             CVPR 2022 Oral             |

## Review

| Year                                                                                             |                                        Title&Project Page                                        |                         Brief Description                         |      Conf/Jour      |
| ------------------------------------------------------------------------------------------------ |:------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------:|:-------------------:|
| [2023](/NeRF/Review/A%20Critical%20Analysis%20of%20NeRF-Based%203D%20Reconstruction)             | [A Critical Analysis of NeRF-Based 3D Reconstruction](https://www.mdpi.com/2072-4292/15/14/3585) | 对比了Colmap摄影测量法和NeRF-based方法在3D Reconstruction中的优劣 | MDPI remote sensing |
| [2023](/NeRF/Review/NeRF%20in%20the%20industrial%20and%20robotics%20domain)                      |                 [Maftej/iisnerf (github.com)](https://github.com/Maftej/iisnerf)                 |                探索了NeRF在工业和机器人领域的应用                 |        None         |
| [2023](/NeRF/Review/2023%20Conf%20about%20NeRF)                                                  |                                               None                                               |                   2023年NeRF Review ref others                    |        None         |
| [2023](/NeRF/Review/A%20Review%20of%20Deep%20Learning-Powered%20Mesh%20Reconstruction%20Methods) |                  A Review of Deep Learning-Powered Mesh Reconstruction Methods                   |       介绍了几种3D模型表示方法+回顾了将DL应用到Mesh重建中的方法       |        None         |

## Sampling

加速训练和渲染过程、提高新视图质量

| Year                                        | Title&Project Page                                                                                                              | Brief Description                                          |          Conf/Jour           |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |:----------------------------:|
| [2023](/NeRF/Sampling/NerfAcc)              | [NerfAcc Documentation — nerfacc 0.5.3](https://www.nerfacc.com/en/latest/)                                                     | 一种新的采样方法可以加速NeRF                               |            arXiv             |
| [2021](/NeRF/Sampling/Mip-NeRF)             | [mip-NeRF ](https://jonbarron.info/mipnerf/)                                                                                    | 截头圆锥体采样方法+IPEncoding                              |             ICCV             |
| [2023](/NeRF/Sampling/Floaters%20No%20More) | [Floaters No More: Radiance Field Gradient Scaling for Improved Near-Camera Training](https://gradient-scaling.github.io/#Code) | 通过梯度缩放，解决基于NeRF重建场景中的背景塌陷和镜前漂浮物 | The Eurographics Association |

- [Improving Neural Radiance Fields Using Near-Surface Sampling With Point Cloud Generation (arxiv.org)](https://arxiv.org/pdf/2310.04152.pdf)

## Sparse images/Generalization

| Year                            | Title&Project Page                                                                                                    | Brief Description                                             |    Conf/Jour     |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |:----------------:|
| [2022](/NeRF/Sparse/SparseNeuS) | [SparseNeuS: Fast Generalizable Neural Surface Reconstruction from Sparse Views](https://www.xxlong.site/SparseNeuS/) | 多层几何推理框架+多尺度颜色混合+一致性感知的微调              |       ECCV       |
| 2023                            | [SparseNeRF](https://sparsenerf.github.io/)                                                                           | 利用来自现实世界不准确观测的深度先验知识                      | Technical Report |
| 2021                            | [pixelNeRF: Neural Radiance Fields from One or Few Images (alexyu.net)](https://alexyu.net/pixelnerf/)                |                                                               |       CVPR       |
| [2022](/NeRF/Sparse/FORGE)      | [FORGE (ut-austin-rpl.github.io)](https://ut-austin-rpl.github.io/FORGE/)                                             | voxel特征提取2D-->3D+相机姿态估计+特征共享和融合+神经隐式重建 |      ArXiv       |
| [2023](/NeRF/Sparse/FreeNeRF)   | [FreeNeRF: Frequency-regularized NeRF (jiawei-yang.github.io)](https://jiawei-yang.github.io/FreeNeRF/)               | 稀疏视图训练时，逐步开放高频分量可以获得更好的效果，遮挡正则消除floaters                                                              |       CVPR       |

## Surface Reconstruction

| Year                                                      |                                                                    Title&Project Page                                                                    |                                           Brief Description                                           |                               Conf/Jour                               |
| --------------------------------------------------------- |:--------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------:|
| [2021](/NeRF/SurfaceReconstruction/Neus)                  |         [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction](https://lingjie0206.github.io/papers/NeuS/)          |                                         Neus: SDF表面重建方法                                         |                                NeurIPS                                |
| [2022](/NeRF/SurfaceReconstruction/Instant-NSR)           |                         [Human Performance Modeling and Rendering via Neural Animated Mesh](https://zhaofuq.github.io/NeuralAM/)                         |                                  NSR: Neus_TSDF + NGP，但是依赖mask                                   |                             SIGGRAPH Asia                             |
| [2023](/NeRF/SurfaceReconstruction/Neus-Instant-nsr-pl)   |                                          [bennyguo/instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl)                                           |                                        Neus+NeRF+Nerfacc+tcnn                                         |                                 None                                  |
| [2023](/NeRF/SurfaceReconstruction/Neuralangelo)          |                     [Neuralangelo: High-Fidelity Neural Surface Reconstruction](https://research.nvidia.com/labs/dir/neuralangelo/)                      |                                       NGP_but数值梯度+Neus_SDF                                        | IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**) |
| [2023](/NeRF/SurfaceReconstruction/PermutoSDF)            |                                                [PermutoSDF](https://radualexandru.github.io/permuto_sdf/)                                                |       NGP_butPermutohedral lattice + Neus_SDF，曲率损失和颜色MLP正则解决镜面+无纹理区域，更光滑       |                      IEEE/CVF Conference on CVPR                      |
| [2023](/NeRF/SurfaceReconstruction/NeuDA)                 |                                                    [NeuDA](https://3d-front-future.github.io/neuda/)                                                     |                                 NGP_butDeformable Anchors+HPE + Neus                                  |                                 CVPR                                  |
| [2023](/NeRF/SurfaceReconstruction/Shadow&Highlight/NeRO) |             [NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images](https://liuyuan-pal.github.io/NeRO/)             | Neus_SDF 新的光表示方法可以重建准确的几何和BRDF，但是细节处由于太光滑而忽略，反射颜色也依赖准确的法线 |                          SIGGRAPH (ACM TOG)                           |
| [2021](/NeRF/SurfaceReconstruction/UNISURF)               |           [UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction ](https://moechsle.github.io/unisurf/)           |                            UNISURF用占用值来表示表面，代替NeRF中的$\alpha$                            |                              ICCV (oral)                              |
| [2023](/NeRF/SurfaceReconstruction/PlankAssembly)         |  [PlankAssembly: Robust 3D Reconstruction from Three Orthographic Views with Learnt Shape Programs](https://manycore-research.github.io/PlankAssembly/)  |                  基于Transform的自注意力提出模型,将2D三视图转化成3D模型的代码形式DSL                  |                                 ICCV                                  |
| [2023](/NeRF/SurfaceReconstruction/NeUDF)                 |                                                       [NeUDF](http://geometrylearning.com/neudf/)                                                        |                          使用UDF，可以重建具有任意拓扑的表面，例如非水密表面                          |                                 CVPR                                  |
| [2023](/NeRF/SurfaceReconstruction/NeuS2)                 |              [NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-view Reconstruction](https://vcai.mpi-inf.mpg.de/projects/NeuS2/)               |                          基于Neus、NGP和NSR，实现高质量快速的静态和动态建模                           |                                 ICCV                                  |
| [2023](/NeRF/SurfaceReconstruction/Color-NeuS)            |                                 [Color-NeuS (colmar-zlicheng.github.io)](https://colmar-zlicheng.github.io/color_neus/)                                  |                         解决了类Neus方法推理时表面颜色提取困难和不正确的问题                          |                                 arXiv                                 |
| [2022](/NeRF/SurfaceReconstruction/HF-NeuS)               |                        [HF-NeuS: Improved Surface Reconstruction Using High-Frequency Details](https://github.com/yiqun-wang/HFS)                        |                新的SDF与透明度$\alpha$关系函数,将SDF分解为基和位移两个独立隐函数的组合                |                                NeurIPS                                |
| [2022](/NeRF/SurfaceReconstruction/Geo-Neus)              |            [Geo-Neus: Geometry-Consistent Neural Implicit Surfaces Learning for Multi-view Reconstruction](https://github.com/GhiXu/Geo-Neus)            |        使用COLMAP产生的稀疏点来作为SDF的显示监督,具有多视图立体约束的隐式曲面上的几何一致监督         |                                NeurIPS                                |
| [2020](/NeRF/SurfaceReconstruction/IDR)                   |        [Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance (lioryariv.github.io)](https://lioryariv.github.io/idr/)        |               端到端的IDR：可以从masked的2D图像中学习3D几何、外观，*允许粗略的相机估计*               |                                NeurIPS                                |
| [2023](/NeRF/SurfaceReconstruction/FlexiCubes)            | [Flexible Isosurface Extraction for Gradient-Based Mesh Optimization (FlexiCubes) (nvidia.com)](https://research.nvidia.com/labs/toronto-ai/flexicubes/) |                                      一种新的Marching Cube的方法                                      |                 ACM Trans. on Graph. (SIGGRAPH 2023)                  |
|                                                           |                                                                                                                                                          |                                                                                                       |                                                                       |

## Shadow&Highlight

| Year                                                            | Title&Project Page                                                                                                               | Brief Description                                                                                          |     Conf/Jour      |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |:------------------:|
| 2023                                                            | [Relighting Neural Radiance Fields with Shadow and Highlight Hints](https://nrhints.github.io/)                                  | 数据集使用相机位姿和灯源位姿,训练集大约500张,Shadow and Highlight hints                                    |      SIGGRAPH      |
| [2023](/NeRF/SurfaceReconstruction/Shadow&Highlight/NeRO)       | [NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images](https://liuyuan-pal.github.io/NeRO/) | Neus_SDF 新的光表示方法可以重建准确的几何和BRDF，但是细节处由于太光滑而忽略，反射颜色也依赖准确的法线      | SIGGRAPH (ACM TOG) |
| [2023](/NeRF/SurfaceReconstruction/Shadow&Highlight/ShadowNeuS) | [ShadowNeuS (gerwang.github.io)](https://gerwang.github.io/shadowneus/)                                                          | 多光照下单视图重建SDF+RGB图像重建外观+BRDF                                                                 |        CVPR        |
| [2022](/NeRF/SurfaceReconstruction/Shadow&Highlight/Ref-NeRF)   | [Ref-NeRF (dorverbin.github.io)](https://dorverbin.github.io/refnerf/)                                                           | 基于球面谐波的IDE编码+预测表面法向+BRDF                                                                    |        CVPR        |
| [2021](/NeRF/SurfaceReconstruction/Shadow&Highlight/NeRFactor)  | [NeRFactor (xiuming.info)](https://xiuming.info/projects/nerfactor/)                                                             | NeRFactor在未知光照条件下从图像中恢复物体形状和反射率                                                      |      SIGGRAPH      |
| [2023](/NeRF/SurfaceReconstruction/Shadow&Highlight/Ref-NeuS)   | [Ref-NeuS (g3956.github.io)](https://g3956.github.io/)                                                                           | Anomaly Detection for Reflection Score + Visibility Identification for Reflection Score+反射感知的光度损失 |     ICCV Oral      |
| [2023](/NeRF/SurfaceReconstruction/Shadow&Highlight/NeuFace)    | [NeuFace: Realistic 3D Neural Face Rendering from Multi-view Images](https://github.com/aejion/NeuFace)                          | **BRDF+SDF+PBR**框架，端到端训练，重建人脸的几何+外观                                                      |        CVPR        |

## Framework

| Year                                  | Title&Project Page                                         | Brief Description |   Conf/Jour  |
| ------------------------------------- | ---------------------------------------------------------- | ----------------- | :----------: |
| [2023](/NeRF/NeRF-Studio) | [nerfstudio](https://docs.nerf.studio/en/latest/)          | 集成现有的NeRF方法       | ACM SIGGRAPH |
| 2022                                  | [SDFStudio](https://autonomousvision.github.io/sdfstudio/) | 集成基于SDF的NeRF方法    |     None     |

# Dalao

| PhD.School                 |                                                     Homepage                                                      |                        Paper                         |
| -------------------------- |:-----------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------:|
| Zhejiang University        |                        [Zhaopeng Cui (zju.edu.cn)](http://www.cad.zju.edu.cn/home/zhpcui/)                        | Global Structure-from-Motion by Similarity Averaging |
| HKU                        |                 [刘缘Yuan Liu - Homepage (liuyuan-pal.github.io)](https://liuyuan-pal.github.io/)                 |                      Neus,NeRO                       |
| CUHK                       |                      [胡文博HU, Wenbo's Homepage (wbhu.github.io)](https://wbhu.github.io/)                       |                      Tri-MipRF                       |
| HKU                        | [Peng Wang (王鹏) (notion.site)](https://quartz-khaan-c6f.notion.site/Peng-Wang-0ab0a2521ecf40f5836581770c14219c) |                         Neus                         |
| UC Berkeley                |                                       [Jon Barron](https://jonbarron.info/)                                       |        Mip-NeRF,Mip-NeRF360,Zip-NeRF,Ref-NeRF        |
| UC Berkeley                |                                 [Matthew Tancik](https://www.matthewtancik.com/)                                  |        NerfStudio,NerfAcc,Plenoxels,Mip-NeRF,        |
| UC Berkeley                |                           [Ben Mildenhall (bmild.github.io)](https://bmild.github.io/)                            |                         NeRF                         |
| ShanghaiTech University    |                      [陈安沛Anpei Chen (apchenstu.github.io)](https://apchenstu.github.io/)                       |                       TensoRF                        |
| University of Pennsylvania |                       [Lingjie Liu (lingjie0206.github.io)](https://lingjie0206.github.io/)                       | NeuS,NeuS2,NeuralUDF,Drag Your GAN                                                     |


## 有趣的应用

| Year                               |                                                                Title&Project Page                                                                 |          Brief Description           | Conf/Jour        |
| ---------------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------:| ---------------- |
| 2023                               |                                     [Seeing the World through Your Eyes](https://world-from-eyes.github.io/)                                      |        从人眼的倒影中重建物体        | None             |
| [2023](/NeRF/Interesting/PAniC-3D) | [PAniC-3D Stylized Single-view 3D Reconstruction from Portraits of Anime Characters](https://github.com/shuhongchen/panic3d-anime-reconstruction) |      从插画风格角色肖像中重建3D      | CVPR             |
| 2023                               |                                          [LERF: Language Embedded Radiance Fields](https://www.lerf.io/)                                          | 用语言查询空间中的3D物体，并高亮显示 | ICCV 2023 (Oral) |
