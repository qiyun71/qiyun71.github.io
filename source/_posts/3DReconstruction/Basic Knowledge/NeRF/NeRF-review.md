---
title: NeRF目前进展
date: 2023-06-25T13:01:00.000Z
tags:
  - NeRF
  - Review
categories: 3DReconstruction/Basic Knowledge/NeRF
date updated: 2023-08-11T17:30:59.000Z
---

NeRF相关的论文 at CVPR/ICCV/ECCV/NIPS/ICML/ICLR/SIGGRAPH
[计算机视觉顶会2022截稿时间及会议时间_ijcai2024截稿日期-CSDN博客](https://blog.csdn.net/weixin_43962054/article/details/121762182)

| My post                                                                                                                          | Brief description              | status            |
| -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ----------------- |
| [NeRF](NeRF-Principle.md) + [Code](NeRF-code.md)                                                                           | NeRF 原理 + 代码理解           | Completed         |
| [NeuS](NeuS.md) + [Code](Neus-code.md)                                         | 表面重建方法 SDFNetwork        | Completed         |
| [InstantNGP](NeRF-InstantNGP.md) + [Tiny-cuda-nn](NeRF-InstantNGP-code.md)                           | 加速 NeRF 的训练和推理         | Completed（Tcnn） |
| [Instant-nsr-pl](Neus-Instant-nsr-pl.md) + [Code](Neus-Instant-nsr-pl-code.md) | Neus+Tcnn+NSR+pl               | Completed         |
| [Instant-NSR](Instant-NSR.md) + [Code](Instant-NSR-code.md)                    | 快速表面重建                   | Completed         |
| [NeRO](NeRO.md) + [Code](NeRO-code.md)       | 考虑镜面和漫反射的体渲染函数   | In Processing     |
| [NeRF-Mine](NeRF-Mine.md)                                                                                                     | 基于 Instant-nsr-pl 创建的项目 | Completed         |

Related link : [3D Reconstruction](https://paperswithcode.com/task/3d-reconstruction) | [awesome-NeRF-papers](https://github.com/lif314/awesome-NeRF-papers)

<!-- more -->

# NeRF

ECCV 2020 Oral - Best Paper Honorable Mention

| Year                                     |                                              Title&Project Page                                             | Brief Description | Conf/Jour |
| ---------------------------------------- | :---------------------------------------------------------------------------------------------------------: | :---------------: | :-------: |
| [2020](NeRF-Principle.md) | [NeRF:Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf) |        初始文        |    ECCV   |


# 原理改进

Volume Rendering Function

### PL-NeRF
[mikacuy/PL-NeRF: NeRF Revisited: Fixing Quadrature Instability in Volume Rendering, Neurips 2023 (github.com)](https://github.com/mikacuy/PL-NeRF)

NeRF 分段常数积分 --> PL-NeRF 分段线性积分
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures20231101151141.png)


- StEik：基于 SDF 的隐式场优化问题(Neural SDF)，提出一个新的约束项
  - [NeurIPS 2023 | 三维重建中的Neural SDF(Neural Implicit Surface) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/649921965)
  - [sunyx523/StEik (github.com)](https://github.com/sunyx523/StEik)
- Rethinking Directional Integration in Neural Radiance Fields，修改了 **NeRF 的渲染方程**
  - [Rethinking Directional Integration in Neural Radiance Fields (arxiv.org)](https://arxiv.org/abs/2311.16504)

## Camera Pose


| Year | Title&Project Page                                                                                                                                                    | Brief Description                | Conf/Jour |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |:---------:|
| 2023 | [[2311.17119] Continuous Pose for Monocular Cameras in Neural Implicit Representation (arxiv.org)](https://arxiv.org/abs/2311.17119)                                  | 将单目相机姿势优化为时间连续函数 |           |
| 2023 | [Pose-Free Generalizable Rendering Transformer (zhiwenfan.github.io)](https://zhiwenfan.github.io/PF-GRT/)                                                            |                                  |           |
| 2023 | [[2310.02687] USB-NeRF: Unrolling Shutter Bundle Adjusted Neural Radiance Fields (arxiv.org)](https://arxiv.org/abs/2310.02687)                                       |                                  |           |
| 2023 | [[2309.11326] How to turn your camera into a perfect pinhole model (arxiv.org)](https://arxiv.org/abs/2309.11326)                                                     |                                  |           |
| 2023 | [[2312.08760] CF-NeRF: Camera Parameter Free Neural Radiance Fields with Incremental Learning (arxiv.org)](https://arxiv.org/abs/2312.08760)                          |                                  |   AAAI    |
| 2023 | [[2312.15238] NoPose-NeuS: Jointly Optimizing Camera Poses with Neural Implicit Surfaces for Multi-view Reconstruction (arxiv.org)](https://arxiv.org/abs/2312.15238) |                                  |           |

### Other paper about camera pose

- How to turn your camera into a perfect pinhole model

## Image问题

| Year | Title&Project Page                                                                                                                                                               | Brief Description | Conf/Jour |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |:---------:|
| 2024 | [[2401.00825] Sharp-NeRF: Grid-based Fast Deblurring Neural Radiance Fields Using Sharpness Prior (arxiv.org)](https://arxiv.org/abs/2401.00825)                                 |                   |   WACV    |
| 2023 | [[2312.15942] Pano-NeRF: Synthesizing High Dynamic Range Novel Views with Geometry from Sparse Low Dynamic Range Panoramic Images (arxiv.org)](https://arxiv.org/abs/2312.15942) | HDR                  |           |
| 2024 | [[2401.03257] RustNeRF: Robust Neural Radiance Field with Low-Quality Images (arxiv.org)](https://arxiv.org/abs/2401.03257)                                                      | 低质量图像        |           |
| 2024     | [Fast High Dynamic Range Radiance Fields for Dynamic Scenes (guanjunwu.github.io)](https://guanjunwu.github.io/HDR-HexPlane/)                                                                                                                                                                                 | Fast HDR          |           |



## Efficiency

### Encoding

| Year                                                    |                                                                 Title&Project Page                                                                  |                                     Brief Description                                     |                               Conf/Jour                               |
| ------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|:---------------------------------------------------------------------:|
| [2022](NeRF-InstantNGP.md)                |                  [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://nvlabs.github.io/instant-ngp/)                   |                                     多分辨率哈希编码                                      |                ACM Transactions on Graphics (SIGGRAPH)                |
| [2023](Neus-Instant-nsr-pl.md) |                                        [bennyguo/instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl)                                        |                                  Neus+NeRF+Nerfacc+tcnn                                   |                                 None                                  |
| [2023](Neuralangelo.md)        |                   [Neuralangelo: High-Fidelity Neural Surface Reconstruction](https://research.nvidia.com/labs/dir/neuralangelo/)                   |                                 NGP_but数值梯度+Neus_SDF                                  | IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**) |
| [2023](PermutoSDF.md)          |                                             [PermutoSDF](https://radualexandru.github.io/permuto_sdf/)                                              | NGP_butPermutohedral lattice + Neus_SDF，曲率损失和颜色MLP正则解决镜面+无纹理区域，更光滑 |                      IEEE/CVF Conference on CVPR                      |
| [2023](NeuDA.md)               |                                                  [NeuDA](https://3d-front-future.github.io/neuda/)                                                  |                           NGP_butDeformable Anchors+HPE + Neus                            |                                 CVPR                                  |
| [2023](Zip-NeRF.md)                       |                                            [Zip-NeRF (jonbarron.info)](https://jonbarron.info/zipnerf/)                                             |                                     NGP + Mip-NeRF360                                     |                                 ICCV                                  |
| [2023](Tri-MipRF.md)                                    | [Tri-MipRF: Tri-Mip Representation for Efficient Anti-Aliasing Neural Radiance Fields (wbhu.github.io)](https://wbhu.github.io/projects/Tri-MipRF/) |                Tri-MipRF encoding(TensoRF + NGP)+ Cone Sampling(Mip-NeRF)                 |                                 ICCV                                  |
| [2022](TensoRF.md)                        |                          [TensoRF: Tensorial Radiance Fields (apchenstu.github.io)](https://apchenstu.github.io/TensoRF/)                           |                 TensoRF引入VM分解，提高重建的质量和速度，并减小了内存占用                 |                                 ECCV                                  |
| [2023](Strivec.md)                        |                                   [Zerg-Overmind/Strivec (github.com)](https://github.com/Zerg-Overmind/Strivec)                                    |                    局部张量CP分解为三向量+多尺度网格+占用网格采样方式                     |                                 ICCV                                  |
| [2023](3D%20Gaussian%20Splatting.md)   |        [3D Gaussian Splatting for Real-Time Radiance Field Rendering (inria.fr)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)         |            优化3D高斯点云，并使用splatting渲染方式，实现实时高分辨率训练+渲染             |                               SIGGRAPH                                |
| [2023](BakedSDF.md)                       |                                                       [BakedSDF](https://bakedsdf.github.io/)                                                       |                   VolSDF+MarchingCube得到mesh，漫反射+高斯叶来渲染颜色                    |                               SIGGRAPH                                |
|                                                         |                                                                                                                                                     |                                                                                           |                                                                       |

## Large Scale Scene

| Year                                                     | Title&Project Page                                                                                            | Brief Description                      | Conf/Jour |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------------- | :-------: |
| [2020](NeRF++.md)         | [nerfplusplus: improves over nerf in 360 capture of unbounded scenes](https://github.com/Kai-46/nerfplusplus) | 将背景的采样点表示为四维向量，与前景分别使用不同的MLP进行训练       |   arXiv   |
| [2022](Mip-NeRF%20360.md) | [mip-NeRF 360](https://jonbarron.info/mipnerf360/)                                                            | 将单位球外的背景参数化，小型提议网络进行精采样，正则化dist消除小区域云雾 |    CVPR   |

## PointCloud

| Year                                                  | Title&Project Page                                                                                                                                    | Brief Description                                                  |               Conf/Jour                |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |:--------------------------------------:|
| [2023](3D%20Gaussian%20Splatting.md) | [3D Gaussian Splatting for Real-Time Radiance Field Rendering (inria.fr)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)                  | 优化3D高斯点云，并使用splatting渲染方式，实现实时高分辨率训练+渲染 |                SIGGRAPH                |
| 2023                                                  | [Globally Consistent Normal Orientation for Point Clouds by Regularizing the Winding-Number Field](https://xrvitd.github.io/Projects/GCNO/index.html) | 使得稀疏点云和薄壁点云法向量一致                                   | ACM Transactions on Graphics(SIGGRAPH) |
| 2023                                                  | [Neural Kernel Surface Reconstruction](https://research.nvidia.com/labs/toronto-ai/NKSR/)                                                             | 从点云中进行表面重建                                               |          CVPR 2023 Highlight           |
| [2022](Point-NeRF.md)                   | [Point-NeRF: Point-based Neural Radiance Fields](https://xharlie.github.io/projects/project_sites/pointnerf/)                                         | 生成初始化点云，基于点云进行增删和体渲染                                                                   |             CVPR 2022 Oral             |

## Review

| Year                                                                                             |                                        Title&Project Page                                        |                         Brief Description                         |      Conf/Jour      |
| ------------------------------------------------------------------------------------------------ |:------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------:|:-------------------:|
| [2023](A%20Critical%20Analysis%20of%20NeRF-Based%203D%20Reconstruction.md)             | [A Critical Analysis of NeRF-Based 3D Reconstruction](https://www.mdpi.com/2072-4292/15/14/3585) | 对比了Colmap摄影测量法和NeRF-based方法在3D Reconstruction中的优劣 | MDPI remote sensing |
| [2023](NeRF%20in%20the%20industrial%20and%20robotics%20domain.md)                      |                 [Maftej/iisnerf (github.com)](https://github.com/Maftej/iisnerf)                 |                探索了NeRF在工业和机器人领域的应用                 |        None         |
| [2023](2023%20Conf%20about%20NeRF.md)                                                  |                                               None                                               |                   2023年NeRF Review ref others                    |        None         |
| [2023](A%20Review%20of%20Deep%20Learning-Powered%20Mesh%20Reconstruction%20Methods.md) |                  A Review of Deep Learning-Powered Mesh Reconstruction Methods                   |       介绍了几种3D模型表示方法+回顾了将DL应用到Mesh重建中的方法       |        None         |

## Sampling

加速训练和渲染过程、提高新视图质量

| Year                            | Title&Project Page                                                                                                                                                                  | Brief Description                                          |          Conf/Jour           |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |:----------------------------:|
| [2023](NerfAcc.md)              | [NerfAcc Documentation — nerfacc 0.5.3](https://www.nerfacc.com/en/latest/)                                                                                                         | 一种新的采样方法可以加速NeRF                               |            arXiv             |
| [2021](Mip-NeRF.md)             | [mip-NeRF ](https://jonbarron.info/mipnerf/)                                                                                                                                        | 截头圆锥体采样方法+IPEncoding                              |             ICCV             |
| [2023](Floaters%20No%20More.md) | [Floaters No More: Radiance Field Gradient Scaling for Improved Near-Camera Training](https://gradient-scaling.github.io/#Code)                                                     | 通过梯度缩放，解决基于NeRF重建场景中的背景塌陷和镜前漂浮物 | The Eurographics Association |
| 2023                            | [ProNeRF: Learning Efficient Projection-Aware Ray Sampling for Fine-Grained Implicit Neural Radiance Fields (kaist-viclab.github.io)](https://kaist-viclab.github.io/pronerf-site/) | 投影感知采样（PAS）                                        |                              |

- [Improving Neural Radiance Fields Using Near-Surface Sampling With Point Cloud Generation (arxiv.org)](https://arxiv.org/pdf/2310.04152.pdf)
- [L0-Sampler: An L0 Model Guided Volume Sampling for NeRF](https://arxiv.org/abs/2311.07044) 分段常数采样改为分段指数采样



## Sparse images/Generalization

| Year                  | Title&Project Page                                                                                                                                                                   | Brief Description                                                        |    Conf/Jour     |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |:----------------:|
| [2022](SparseNeuS.md) | [SparseNeuS: Fast Generalizable Neural Surface Reconstruction from Sparse Views](https://www.xxlong.site/SparseNeuS/)                                                                | 多层几何推理框架+多尺度颜色混合+一致性感知的微调                         |       ECCV       |
| 2023                  | [SparseNeRF](https://sparsenerf.github.io/)                                                                                                                                          | 利用来自现实世界不准确观测的深度先验知识                                 | Technical Report |
| 2021                  | [pixelNeRF: Neural Radiance Fields from One or Few Images (alexyu.net)](https://alexyu.net/pixelnerf/)                                                                               |                                                                          |       CVPR       |
| [2022](FORGE.md)      | [FORGE (ut-austin-rpl.github.io)](https://ut-austin-rpl.github.io/FORGE/)                                                                                                            | voxel特征提取2D-->3D+相机姿态估计+特征共享和融合+神经隐式重建            |      ArXiv       |
| [2023](FreeNeRF.md)   | [FreeNeRF: Frequency-regularized NeRF (jiawei-yang.github.io)](https://jiawei-yang.github.io/FreeNeRF/)                                                                              | 稀疏视图训练时，逐步开放高频分量可以获得更好的效果，遮挡正则消除floaters |       CVPR       |
| 2023                  | [ZeroRF (sarahweiii.github.io)](https://sarahweiii.github.io/zerorf/)                                                                                                                |                                                                          |                  |
| 2023                  | [eezkni/ColNeRF: [AAAI2024] Pytorch implementation of "ColNeRF: Collaboration for Generalizable Sparse Input Neural Radiance Field" (github.com)](https://github.com/eezkni/ColNeRF) |                                                                          |     AAAI2024     |
| 2023                  | [NeuSurf - Project Page (alvin528.github.io)](https://alvin528.github.io/NeuSurf/)                                                                                                                                                                                     |                                                                          |                  |


## Surface Reconstruction

| Year | Title&Project Page | Brief Description | Conf/Jour |
| ---- | :--: | :--: | :--: |
| [2021](NeuS.md) | [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction](https://lingjie0206.github.io/papers/NeuS/) | Neus: SDF表面重建方法 | NeurIPS |
| [2022](Instant-NSR.md) | [Human Performance Modeling and Rendering via Neural Animated Mesh](https://zhaofuq.github.io/NeuralAM/) | NSR: Neus_TSDF + NGP，但是依赖mask | SIGGRAPH Asia |
| [2023](Neus-Instant-nsr-pl.md) | [bennyguo/instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl) | Neus+NeRF+Nerfacc+tcnn | None |
| [2023](Neuralangelo.md) | [Neuralangelo: High-Fidelity Neural Surface Reconstruction](https://research.nvidia.com/labs/dir/neuralangelo/) | NGP_but数值梯度+Neus_SDF | IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**) |
| [2023](PermutoSDF.md) | [PermutoSDF](https://radualexandru.github.io/permuto_sdf/) | NGP_butPermutohedral lattice + Neus_SDF，曲率损失和颜色MLP正则解决镜面+无纹理区域，更光滑 | IEEE/CVF Conference on CVPR |
| [2023](NeuDA.md) | [NeuDA](https://3d-front-future.github.io/neuda/) | NGP_butDeformable Anchors+HPE + Neus | CVPR |
| [2023](NeRO.md) | [NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images](https://liuyuan-pal.github.io/NeRO/) | Neus_SDF 新的光表示方法可以重建准确的几何和BRDF，但是细节处由于太光滑而忽略，反射颜色也依赖准确的法线 | SIGGRAPH (ACM TOG) |
| [2021](UNISURF.md) | [UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction ](https://moechsle.github.io/unisurf/) | UNISURF用占用值来表示表面，代替NeRF中的$\alpha$ | ICCV (oral) |
| [2023](PlankAssembly.md) | [PlankAssembly: Robust 3D Reconstruction from Three Orthographic Views with Learnt Shape Programs](https://manycore-research.github.io/PlankAssembly/) | 基于Transform的自注意力提出模型,将2D三视图转化成3D模型的代码形式DSL | ICCV |
| [2023](NeUDF.md) | [NeUDF](http://geometrylearning.com/neudf/) | 使用UDF，可以重建具有任意拓扑的表面，例如非水密表面 | CVPR |
| [2023](NeuS2.md) | [NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-view Reconstruction](https://vcai.mpi-inf.mpg.de/projects/NeuS2/) | 基于Neus、NGP和NSR，实现高质量快速的静态和动态建模 | ICCV |
| [2023](Color-NeuS.md) | [Color-NeuS (colmar-zlicheng.github.io)](https://colmar-zlicheng.github.io/color_neus/) | 解决了类Neus方法推理时表面颜色提取困难和不正确的问题 | arXiv |
| [2022](HF-NeuS.md) | [HF-NeuS: Improved Surface Reconstruction Using High-Frequency Details](https://github.com/yiqun-wang/HFS) | 新的SDF与透明度$\alpha$关系函数,将SDF分解为基和位移两个独立隐函数的组合 | NeurIPS |
| [2022](Geo-Neus.md) | [Geo-Neus: Geometry-Consistent Neural Implicit Surfaces Learning for Multi-view Reconstruction](https://github.com/GhiXu/Geo-Neus) | 使用COLMAP产生的稀疏点来作为SDF的显示监督,具有多视图立体约束的隐式曲面上的几何一致监督 | NeurIPS |
| [2020](IDR.md) | [Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance (lioryariv.github.io)](https://lioryariv.github.io/idr/) | 端到端的IDR：可以从masked的2D图像中学习3D几何、外观，*允许粗略的相机估计* | NeurIPS |
| [2023](FlexiCubes.md) | [Flexible Isosurface Extraction for Gradient-Based Mesh Optimization (FlexiCubes) (nvidia.com)](https://research.nvidia.com/labs/toronto-ai/flexicubes/) | 一种新的Marching Cube的方法 | ACM Trans. on Graph. (SIGGRAPH 2023) |

## 额外监督

| Year |                                                                         Title&Project Page                                                                          | Brief Description | Conf/Jour |
| ---- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------:|:---------:|
| 2024 | [[2401.03412] N$^{3}$-Mapping: Normal Guided Neural Non-Projective Signed Distance Fields for Large-scale 3D Mapping (arxiv.org)](https://arxiv.org/abs/2401.03412) |  不同的法向监督   |           |

## Shadow&Highlight

| Year                  | Title&Project Page                                                                                                                                                | Brief Description                                                                                          |     Conf/Jour      |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |:------------------:|
| 2023                  | [Relighting Neural Radiance Fields with Shadow and Highlight Hints](https://nrhints.github.io/)                                                                   | 数据集使用相机位姿和灯源位姿,训练集大约500张,Shadow and Highlight hints                                    |      SIGGRAPH      |
| [2023](NeRO.md)       | [NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images](https://liuyuan-pal.github.io/NeRO/)                                  | Neus_SDF 新的光表示方法可以重建准确的几何和BRDF，但是细节处由于太光滑而忽略，反射颜色也依赖准确的法线      | SIGGRAPH (ACM TOG) |
| [2023](ShadowNeuS.md) | [ShadowNeuS (gerwang.github.io)](https://gerwang.github.io/shadowneus/)                                                                                           | 多光照下单视图重建SDF+RGB图像重建外观+BRDF                                                                 |        CVPR        |
| [2022](Ref-NeRF.md)   | [Ref-NeRF (dorverbin.github.io)](https://dorverbin.github.io/refnerf/)                                                                                            | 基于球面谐波的IDE编码+预测表面法向+BRDF                                                                    |        CVPR        |
| [2021](NeRFactor.md)  | [NeRFactor (xiuming.info)](https://xiuming.info/projects/nerfactor/)                                                                                              | NeRFactor在未知光照条件下从图像中恢复物体形状和反射率                                                      |      SIGGRAPH      |
| [2023](Ref-NeuS.md)   | [Ref-NeuS (g3956.github.io)](https://g3956.github.io/)                                                                                                            | Anomaly Detection for Reflection Score + Visibility Identification for Reflection Score+反射感知的光度损失 |     ICCV Oral      |
| [2023](NeuFace.md)    | [NeuFace: Realistic 3D Neural Face Rendering from Multi-view Images](https://github.com/aejion/NeuFace)                                                           | **BRDF+SDF+PBR**框架，端到端训练，重建人脸的几何+外观                                                      |        CVPR        |
| 2024                  | [cuiziteng/Aleth-NeRF: [AAAI 2024] Aleth-NeRF: Illumination Adaptive NeRF with Concealing Field Assumption (github.com)](https://github.com/cuiziteng/Aleth-NeRF) |                                                                                                            |        AAAI        |
| 2023                  | [[2312.08118] Neural Radiance Fields for Transparent Object Using Visual Hull (arxiv.org)](https://arxiv.org/abs/2312.08118)                                      |                                                                                                            |                    |

## 从隐式场中提取表面

| Year | Title&Project Page | Brief Description | Conf/Jour |
| ---- | ---- | ---- | :--: |
| 2023 | [cong-yi/DualMesh-UDF (github.com)](https://github.com/cong-yi/DualMesh-UDF?tab=readme-ov-file) | Surface Extraction from Neural Unsigned Distance Fields | ICCV  |
|  |  |  |  |


## Framework

| Year                                  | Title&Project Page                                         | Brief Description |   Conf/Jour  |
| ------------------------------------- | ---------------------------------------------------------- | ----------------- | :----------: |
| [2023](NeRF-Studio.md) | [nerfstudio](https://docs.nerf.studio/en/latest/)          | 集成现有的NeRF方法       | ACM SIGGRAPH |
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


# 有趣的应用

| Year                               |                                                                Title&Project Page                                                                 |          Brief Description           | Conf/Jour        |
| ---------------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------:| ---------------- |
| 2023                               |                                     [Seeing the World through Your Eyes](https://world-from-eyes.github.io/)                                      |        从人眼的倒影中重建物体        | None             |
| [2023](PAniC-3D.md) | [PAniC-3D Stylized Single-view 3D Reconstruction from Portraits of Anime Characters](https://github.com/shuhongchen/panic3d-anime-reconstruction) |      从插画风格角色肖像中重建3D      | CVPR             |
| 2023                               |                                          [LERF: Language Embedded Radiance Fields](https://www.lerf.io/)                                          | 用语言查询空间中的3D物体，并高亮显示 | ICCV 2023 (Oral) |

