---
title: Finite Element Model 3D Reconstruction
date: 2024-08-27 21:32:35
tags:
  - 3DReconstruction
categories: 3DReconstruction
---

通过多视图三维重建获得三维模型，用于有限元分析/工业设计指导

<!-- more -->

# IDEA

## 直接重建模型

三维重建出点云模型，然后根据点云得到有限元模型，并对有限元模型进行修正。难点在于：
- **三维重建**：三维重建出来的点云精度要高，要能准确得到重建物体的尺寸(摄影测量)
- 格式转换：点云模型如何转换成有限元模型(例如Patran的.x_t格式文件，或者经过处理直接用点云坐标替换bdf文件中的网格节点坐标)
  - 类似DMTet输入点云来优化mesh模型，
  - Surface mesh to Volume mesh，然后通过从mesh模型中得到体网格模型 
    - 必须首先保证 surface mesh 是水密的 内部结构不能复杂 [wildmeshing/fTetWild: Fast Tetrahedral Meshing in the Wild](https://github.com/wildmeshing/fTetWild) [【三维视觉】TetWild / fTetWild学习：将任意mesh处理成流形水密的mesh-CSDN博客](https://blog.csdn.net/weixin_43693967/article/details/134026594)
    - 然后将水密mesh体网格化[Gmsh: a three-dimensional finite element mesh generator with built-in pre- and post-processing facilities](https://gmsh.info/#Documentation) https://mbarzegary.github.io/2022/06/27/surface-to-volume-mesh-using-gmsh/
- 模型修正：确定的or区间的or随机的，修正出来的有限元仿真响应能否真实反映实际测量的响应特征
- 数据集：
  - 图片数据，要拍摄哪些种类的物体(工况)，要有实际的意义(例如机械结构、土木结构等)
  - 模型修正响应，得到有限元模型后，通过Nastran仿真计算输出响应，响应特征的选取

> [How to create an FE Volume Mesh – nTop](https://support.ntop.com/hc/en-us/articles/360037005234-How-to-create-an-FE-Volume-Mesh) Design Analysis 设计分析 + Topology Optimization 拓扑优化([nTop](https://www.ntop.com/software/products/)软件不错)
> 

## 间接重建模型

首先获得物体的边缘，然后根据CAD或者Solidworks建立准确的三维模型


# 文献调研

| Year | Paper                                                                                                                                                                                                               | Overview                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Important for me                       |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| 2024 | [Surface reconstruction from FE mesh model](https://academic.oup.com/jcde/article/6/2/197/5732318?login=false)<br>                                                                                                  | ![m_j.jcde.2018.05.004-fx1.jpeg (520×297](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/jcde/6/2/10.1016_j.jcde.2018.05.004/4/m_j.jcde.2018.05.004-fx1.jpeg?Expires=1732527016&Signature=Eb9Hsdk5dB0etDgtIp-urXmxXTQgtqLRv1aiQzuPxjU13coGiq~HSSAAtHgWUCMhqBxBDil5xolFmPLIsrDmXtG4AogoqwFl~SHRKUrcdl7QcDpBE3fbp1mHR2Se9pMyOFfjWYu88IUjSrJMx-Z1vIJhCKVL0PwX0kdO81rR5c4AepKjlyEV-lJ3OEOzVP5sxmO9pPH72DdWvxE9sHWdA0foQxkcfU7WVcETuh1g3epYuP7wvRLdUprJQi5~snVCZiqWanqRGE~c4Loh7RydigL3ZBOdBNkzcFWBk833N2iT0yY9yIBaWhjfSjM3eyKOuHJXC-cWaJ0w6g6aTALNew__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA) | 本文提出了一种从FE网格模型中提取自由形状 B 样条曲面和某些特征曲线的方法 |
| 2024 | [3D Neural Edge Reconstruction](https://arxiv.org/pdf/2405.19295)                                                                                                                                                   | ![overview.jpg (1294×699)](https://neural-edge-map.github.io/resources/overview.jpg)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 3D 模型边缘重建                              |
| 2020 | [A Review on Finite Element Modeling and Simulation of the Anterior Cruciate Ligament Reconstruction](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2020.00967/full) | ![fbioe-08-00967-g003.jpg (3353×817)](https://www.frontiersin.org/files/Articles/546485/fbioe-08-00967-HTML/image_m/fbioe-08-00967-g003.jpg)                                                                                                                                                                                                                                                                                                                                                                                                                                                              |                                        |
| 2020 | [Personalized Knee Geometry Modeling Based on Multi-Atlas Segmentation and Mesh Refinement](https://ieeexplore.ieee.org/document/9042322)                                                                           | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241023144706.png)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |                                        |
| 2020 | [SafeACL: Ligament reconstruction based on subject-specific musculoskeletal and finite element models ](https://pasithee.library.upatras.gr/iisa/article/view/3329)                                                 | ![image.png](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241023144838.png)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |                                        |
|      |                                                                                                                                                                                                                     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |                                        |
|      |                                                                                                                                                                                                                     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |                                        |



## CAD 指令流

| Year | Paper                                                                                                                                                                                        | 研究对象 | 研究内容        | 研究方法                             | Important for me |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ----------- | -------------------------------- | ---------------- |
| 2023 | [3D reconstruction based on hierarchical reinforcement learning with transferability - IOS Press](https://content.iospress.com/articles/integrated-computer-aided-engineering/ica230710)<br> | 3D模型 | 3D重建 CAD 指令 | Reinforcement Learning (RL), CAD | 强化学习、CAD指令学习     |


## 激光雷达扫描仪

| Year         | Paper                                                                                                                                                                                                                                                                                                                                                                                     | 研究对象                                   | 研究内容        | 研究方法                                                       | Important for me                                                                                                                               |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ----------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023         | [Concrete spalling damage detection and seismic performance evaluation for RC shear walls via 3D reconstruction technique and numerical model updating](Concrete%20spalling%20damage%20detection%20and%20seismic%20performance%20evaluation%20for%20RC%20shear%20walls%20via%203D%20reconstruction%20technique%20and%20numerical%20model%20updating.md)<br>Automation in Construction<br> | 钢筋混凝土*RC*剪力墙<br>                       | 损伤检测，抗震性能评估 | Numerical model updating+3D reconstruction(智能手机内置的激光雷达扫描仪) | 通过**信息转移点矩阵**的新概念，建立了被检墙重构的三维点云模型中隐藏的缺陷信息与其对应有限元模型的性能变化之间的映射关系。实验结果表明，所提出的方法能够成功定位混凝土剥落损伤，量化被检测试件的承载力变化<br>创新：之前很少研究混凝土剥落损伤与钢筋混凝土剪力墙残余承载力之间的关系 |
| 2021<br><br> | [A novel intelligent inspection robot with deep stereo vision for three-dimensional concrete damage detection and quantification - Cheng Yuan, Bing Xiong, Xiuquan Li, Xiaohan Sang, Qingzhao Kong, 2022](https://journals.sagepub.com/doi/10.1177/14759217211010238) <br>Structural Health Monitoring                                                                                    | *RC*<br>reinforced concrete structures | 裂纹评估        | 智能检测机器人 with deep stereo vision                            |                                                                                                                                                |
| 2021         | [Automated finite element modeling and analysis of cracked reinforced concrete beams from three dimensional point cloud - Yu - 2021 - Structural Concrete - Wiley Online Library](https://onlinelibrary.wiley.com/doi/10.1002/suco.202100194)<br>Structural Concrete<br>                                                                                                                  | 裂纹RC梁                                  | 自动有限元建模和分析  | 点云三维建模                                                     | 利用收集到的三维点数据自动生成了被检测梁的数值模型                                                                                                                      |
| 2021         | [Automated generation of FE models of cracked RC beams based on 3D point clouds and 2D images Journal of Civil Structural Health Monitoring](https://link.springer.com/article/10.1007/s13349-021-00525-5)<br>Journal Article                                                                                                                                                             | 裂纹RC梁                                  | 自动有限元模型     | 3D点云和2D图像，点云配准算法ICP-based DLT                              | 点云配准算法ICP-based DLT将图像上的裂纹配准到3D点云上，从而自动生成有限元模型                                                                                                 |


| Year | Paper                                                                                                                                                                                                                        | 研究对象                 | 研究内容                                                                                   | 研究方法                                                 | Important for me                                                        |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------- |
| 2022 | [Laser-based finite element model reconstruction for structural mechanics](https://pubs.aip.org/aip/adv/article/12/10/105111/2819776/Laser-based-finite-element-model-reconstruction)<br>北京科技大学土木与资源工程学院城市地下空间工程北京市重点实验室<br> | structural mechanics | finite element model reconstruction, 自动识别损坏部件,The non-destructive evaluation (NDE无损评估) | Laser-based，HLS(手持激光扫描仪型号：Creaform Handyscan 3D 700) | 将来自 3D 激光扫描系统的条件数据转换为能够描述组件机械响应的 FEM。对如桥梁等大型物体，自动识别受损部位并局部更新到有限元模型中<br> |
| 2016 | [Exploitation of Geometric Data provided by Laser Scanning to Create FEM Structural Models of Bridges](https://ascelibrary.org/doi/10.1061/%28ASCE%29CF.1943-5509.0000807)                                                   |                      |                                                                                        |                                                      |                                                                         |


| Year | Paper                                                                                                                                                                                                                 | 研究对象 | 研究内容 | 研究方法 | Important for me |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ---- | ---- | ---------------- |
| 2024 | [Three-dimensional finite element simulation and reconstruction of jointed rock models using CT scanning and photogrammetry - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S167477552300272X)<br> |      |      |      |                  |



## 生成式方法

| Year      | Paper                                                                                  | 研究对象                     | 研究内容                           | 研究方法   | Important for me |
| --------- | -------------------------------------------------------------------------------------- | ------------------------ | ------------------------------ | ------ | ---------------- |
| **2022R** | [Hex-Mesh Generation and Processing: A Survey](https://dl.acm.org/doi/10.1145/3554920) | FE Volume Mesh(Hex-Mesh) | Mesh Generation and Processing | Review | 六面体网格生成技术综述      |
|           |                                                                                        |                          |                                |        |                  |


## 根据三角网格优化体网格

可以无缝地从mesh直接得到hex mesh

| Year | Paper | 研究对象 | 研究内容 | 研究方法 | Important for me |
| ---- | ----- | ---- | ---- | ---- | ---------------- |
| 2024 |       |      |      |      |                  |
|      |       |      |      |      |                  |

- idea from：
  - [Fast and Robust Hexahedral Mesh Optimization via Augmented Lagrangian, L-BFGS, and Line Search | PDF](https://arxiv.org/pdf/2410.11656)
  - [X 上的 Zhenjun Zhao：“Fast and Robust Hexahedral Mesh Optimization via Augmented Lagrangian, L-BFGS, and Line Search Hua Tong, Yongjie Jessica Zhang https://t.co/XAjGcAZxMp https://t.co/yu7zDYbdHY” / X](https://x.com/zhenjun_zhao/status/1846392374425276748)

