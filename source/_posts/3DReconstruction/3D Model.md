---
title: 3D Model
date: 2024-11-27 17:26:47
tags:
  - 3DReconstruction
categories: 3DReconstruction
---

三维模型的各种形式

<!-- more -->

3D Represent

![image.png|555](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241010195323.png)

> [10种最流行的3D文件格式 - BimAnt](http://www.bimant.com/blog/top10-popular-3d-file-formats/)

目前，机械行业中不同的模型形式：
- 构造实体几何[Constructive solid geometry - Wikipedia](https://en.wikipedia.org/wiki/Constructive_solid_geometry)，构造实体几何允许建模者通过使用布尔运算符组合更简单的对象来创建复杂的表面或对象，从而可能通过组合一些原始对象来生成视觉上复杂的对象。(Solidworks、AutoCAD等机械设计的三维模型)
- 多边形网格[Polygon mesh - Wikipedia](https://en.wikipedia.org/wiki/Polygon_mesh)，面通常由三角形（triangle mesh）、四边形（quads）或其他简单凸多边形（n-gons）组成。多边形网格也可能更通常由凹多边形或带有孔的偶数多边形组成。体积网格与多边形网格的不同之处在于，它们显式表示结构的表面和内部区域，而多边形网格仅显式表示表面（体积是隐式的）。(3D打印、CAE仿真)
  - 近似网格编码，3D 模型的表面覆盖有微小多边形（通常是三角形）的网格。 此过程也称为“曲面细分”，因此这些文件格式也称为曲面细分格式。
  - 精确网格编码，精确文件格式使用非均匀有理基样条 (NURBS，一种计算机生成的数学模型）形成的曲面，而不是多边形。对于一些复杂的连接处需要用到样条来表示 
- 箱体建模[Box modeling - Wikipedia](https://en.wikipedia.org/wiki/Box_modeling)，长方体建模是 3D 建模中的一种技术，其中使用基本形状（例如长方体、圆柱体、球体等）来制作最终模型的基本形状。然后使用这个基本形状来雕刻最终模型。该过程使用许多重复的步骤来达到最终产品，这可以导致更高效、更可控的建模过程。(艺术家雕刻过程)

# Solid

目前机械行业使用最多的模型形式

### 获取

人工建模Solidworks、Creo、Rhino...

强化学习重建CAD模型：[3D reconstruction based on hierarchical reinforcement learning with transferabilit](https://content.iospress.com/download/integrated-computer-aided-engineering/ica230710?id=integrated-computer-aided-engineering%2Fica230710#page=4.49)

![image.png|444](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20241127191609.png)

从FE surface mesh model中提取 free-form B-spline surface：[Surface reconstruction from FE mesh model](https://watermark.silverchair.com/j.jcde.2018.05.004.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA18wggNbBgkqhkiG9w0BBwagggNMMIIDSAIBADCCA0EGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMwUnyFsLtbCJdBtjiAgEQgIIDEp9LVU5TBBEEUkKl4THHIULKGCsKlH7EI7pdhfNxxfX3umFNAxqNhmlpCigHdbo7gx0CRKO_NA7al7Fy1LN_nNc8b4OhJseTFj_7XbJ-PyyNl7uZgXUZypV9xgU2ZDrY3B6nxfs8GDsGfL2ralz-H1HGGatgOG6FWENXD6pSOD6-pUp7h7-REU_he6dFVXQ9sQUuNL-vjOESZgOdsJ3l6kFmsLfcKy_jLGbdnnrVkYY-uasnWOSU0btcbstWj6RYa0AXu_gjXf4f54BEKLPh1jMhKlJMaooADaL76r933yPRWJQJ-qGIlqqPNUoGJrJ5GBFNwud2le0y5BRJZQb3xBMFT8DrAKc1sw86ANa1jsuoqdOhSyueemJcVexQTo9ZS8J21YlAyTkQ09nQMpmBMsTwIRQq1nLsx7ZbShkOUj8W-1ocoPGb0H78t_XJcFEuFyJdryIc9skBF2AYGeSfmrromZvXc4Z34X5tVZtdJx29mLhIcrXb7KLdSnYDH0lM8XYt97bI1cceATNUcIFaSWKiIGerFIzCe4uoNrjX2MdaLTX8054um36E5XEWY7y6CvNK7jNjevSbdhJholMYAwuAJs76CuGHiALBg-6l6Qra6dRTC4B3cDalc8aIzMqQ3Sd_qjkJ_Sp315znKRRnJoYPTWLrd2S6y2in5k3KPTxrb7Ijs0-8bsaYbvwRtkWRwx-DW62UwNQHhszOBr0o9iaAxYm2GLTKc-EWp5ZeBfmlTLnEyjzQsjWJl1KwzNxt98o8DEreSjOfogI2mt3kL_QC-GFKYK3SyaWO5NdVcyTDi7dFbGgi1ptl8TVyBKEuPHLuSAEdUo4OK6CLCh8TZido1zMpbp3vZGUquDR2VTNJIRS2kzd9kdf5jKEnZyaLf9qCmEXNCEDn_qmWvtPwPMMfMWfFkVrv9bPsFcnKDVrhdgLA-NqIxbtF3iz03j0MZTRlvxnl1Aeh6Yl1s3jjAM8YJO-A_hAV6M8x0pnalcSnQHbfw5KuxR3J_fTdrRfDBKJ8hQc-ZZZ8DnNzQJAHq6IiZA)

![m_j.jcde.2018.05.004-fx1.jpeg (520×297)](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/jcde/6/2/10.1016_j.jcde.2018.05.004/4/m_j.jcde.2018.05.004-fx1.jpeg?Expires=1734581105&Signature=i7UokZKU8T5xAmC83tU~dAHWaA-7um29yiNNZeaPzn4KaXilULv7WShmgCXG2Uvb0Kmflw5lCCJIjZXo9nzfyv25KMqbCfwn82lMiE0vhUVaQfIQlIpU7L~RqIPQDZynWOEYHv~8v-G65Pd7vumZOcuwA5UU~ZkZbd6WyOfu8e6gxqEB2GwIe9IyriC~cyVQt1R6bEyV6j8QhyZ3E62IGtWR5F-JcJ8-fj2fiuohaDrLRb1DXknn7fcDwtjVgqiARuth9axhnLS-6rGWuqUWqMjeeilknqbbprToQ~yar-rauzMqa44S-kgc7r0LlkS5x~o14GeGzaEa8~WbJ92EZw__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

# Polygon mesh

其中网格(mesh)为各个领域最常用的表示方法，其可以分为surface mesh和volume mesh：
- 表面mesh：通过多个三角形/四边形组成物体表面，可用于3D打印
- 体mesh：通过多个四面体网格tet/六面体网格hex/其他网格组成实体模型，可用于CAE

## surface mesh

### 获取

- 人工建模：通过直接操纵顶点、边和面来创建和编辑模型。常见的手动建模工具有Autodesk 3ds Max、Blender、Maya等，支持多边形建模、细分曲面建模、NURBS建模(非均匀有理B样条曲线和曲面)等多种技术
- 扫描建模：通过3D扫描设备（如激光扫描仪、结构光扫描仪、深度相机）获取实物的点云数据，再通过软件（如Meshlab、CloudCompare、Geomagic）进行点云处理、网格化，生成精确的三维模型
- 根据图片使用算法进行建模(成本低) [Paper About 3D Reconstruction](Paper%20About%203D%20Reconstruction.md)
  - 显示表示，获取点云/体素/深度图等后然后转mesh，或者直接获取mesh
    - COLMAP通过SFM获取相机位姿+**稀疏点云**，并使用MVS来估计深度图并通过depth fusion来获取稠密点云，最后通过screened poisson surface reconstruction获取mesh
    - 3DGS(SuGaR) 高斯体的中心点即为物体点云，然后可以转mesh
  - 隐式表示：IDR/DVR、NeRF(VolSDF、NeuS)使用MC算法从隐式场中提取物体表面mesh
  - ....

### 优化

从低质量的dense mesh/点云中生成高质量人造Mesh(网格规律，缺陷少)：
- [MeshAnything](https://buaacyw.github.io/mesh-anything/)
- [MeshAnything V2](https://buaacyw.github.io/meshanything-v2/)

![teaser.png (6507×6449)|666](https://buaacyw.github.io/meshanything-v2/teaser.png)


### Other

计算3D surface mesh的体积
[math - How to calculate the volume of a 3D mesh object the surface of which is made up triangles - Stack Overflow](https://stackoverflow.com/questions/1406029/how-to-calculate-the-volume-of-a-3d-mesh-object-the-surface-of-which-is-made-up) | [chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf](http://chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf)

## volume mesh

### 获取

有限元网格划分：[Finite Element Model 3D Reconstruction](Finite%20Element%20Model%203D%20Reconstruction.md) | [Anime Image 3D Reconstruction](Anime%20Image%203D%20Reconstruction.md)
- [Gmsh: a three-dimensional finite element mesh generator with built-in pre- and post-processing facilities](https://gmsh.info/)，其也可以将surface mesh转化为volume mesh [Generate volume mesh from a surface mesh using GMSH · Mojtaba Barzegari](https://mbarzegary.github.io/2022/06/27/surface-to-volume-mesh-using-gmsh/)
- [wildmeshing/fTetWild: Fast Tetrahedral Meshing in the Wild](https://github.com/wildmeshing/fTetWild) [yixin-hu.github.io/ftetwild.pdf](https://yixin-hu.github.io/ftetwild.pdf)
- [Engineering, Design, and Simulation Software | nTop | nTop](https://www.ntop.com/software/products/) 商业软件 [How to create an FE Volume Mesh – nTop Support](https://support.ntop.com/hc/en-us/articles/360037005234-How-to-create-an-FE-Volume-Mesh)

surface mesh转volume mesh
- [mdolab/pyhyp: pyHyp generates volume meshes from surface meshes using hyperbolic marching.](https://github.com/mdolab/pyhyp)
- [iso2mesh: a Matlab/Octave-based mesh generator: Home](https://iso2mesh.sourceforge.net/cgi-bin/index.cgi) surface mesh/3D binary/ gray scalevolumetric images(segmented MRI/CT scans)

![iso2mesh_workflow_v2.png (1096×934)|666](https://iso2mesh.sourceforge.net/upload/iso2mesh_workflow_v2.png)

[VMesh: Hybrid Volume-Mesh Representation for Efficient View Synthesis](https://bennyguo.github.io/vmesh/)
这篇文章是使用混合volume mesh表示来高效地生成新试图，**那么用体渲染思路来优化volume mesh可行否**？

生成式方法：
- [Hex-Mesh Generation and Processing: A Survey](https://dl.acm.org/doi/pdf/10.1145/3554920)


### 优化

[Fast and Robust Hexahedral Mesh Optimization via Augmented Lagrangian, L-BFGS, and Line Search | PDF](https://arxiv.org/pdf/2410.11656)
[X 上的 Zhenjun Zhao：“Fast and Robust Hexahedral Mesh Optimization via Augmented Lagrangian, L-BFGS, and Line Search Hua Tong, Yongjie Jessica Zhang https://t.co/XAjGcAZxMp https://t.co/yu7zDYbdHY” / X](https://x.com/zhenjun_zhao/status/1846392374425276748)

