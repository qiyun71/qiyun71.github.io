---
title: Learn-Colmap
date: 2023-08-23 17:00:00
tags:
  - Colmap
categories: Learn
---

[colmap tutorial](https://colmap.github.io/tutorial.html)

更好的重建效果需要拍摄的图片保证:

-   好的纹理
-   相似的照明条件，避免高动态范围场景（例如，逆光阴影或透过门窗的照片），避免在光滑表面上出现反射光
-   高视觉重叠度，确保每个对象至少在 3 张图像中可见，图像越多越好
-   从不同的视角拍摄图像，不要只通过旋转相机来拍摄相同位置的图像

<!-- more -->

Colmap 算法主要包括 SFM 和 MVS

![image.png|500](https://raw.githubusercontent.com/yq010105/Blog_images/main/pictures/20230717204531.png)

# Structure-from-Motion(SFM)

> [Structure-from-Motion Revisited (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=553224841006579712&noteId=1991438575104619008) > [SFM 算法原理初简介 | jiajie (gitee.io)](https://jiajiewu.gitee.io/post/tech/slam-sfm/sfm-intro/)

增量式 SFM：输入一系列从多个不同视角对相同物体 overlapping 的图像，输出稀疏 3D 点云和所有图像对应的相机内外参

1. Feature detection and extraction
2. Feature matching and geometric verification
3. Structure and motion reconstruction

![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231005200141.png)

SFM:

-   Correspondence Search. **输入图像** --> 一组经过几何验证的**图像对**，以及每个点的图像**投影图**
    -   Feature Extraction. SIFT 算子(可以是任何一种特异性较强的特征)
        -   SIFT 算法(Scale-invariant feature transform)是一种电脑视觉的算法用来侦测与描述影像中的局部性特征，它在空间尺度中寻找极值点，并提取出其位置、尺度、旋转不变量，此算法由  David Lowe 在 1999 年所发表，2004 年完善总结, ref: [非常详细的 sift 算法原理解析\_可时间倒数了的博客-CSDN 博客](https://blog.csdn.net/u010440456/article/details/81483145)
        - 改进的SIFT算法，mickey: [Matching 2D Images in 3D: Metric Relative Pose from Metric Correspondences](https://nianticlabs.github.io/mickey/)
    -   Matching.
    -   Geometric Verification. 如果一个有效的变换在图像之间映射了足够数量的特征，它们就被认为是几何验证的
        - 单应性H描述了纯旋转或移动摄像机捕捉平面场景的变换
        - 对极几何通过essential矩阵E(校准)或fundamental矩阵F(未校准)描述了移动摄像机的关系，并且可以使用三焦张量扩展到三个视图
-   Incremental Reconstruction. **scene graph.** --> **pose estimates and reconstructed scene structure as a set of points**
    - Initialization. 选择合适的初始pair至关重要
    - Image Registration. 配准：从度量重建开始，通过使用已配准图像中三角点的特征对应(2D-3D对应)来解决Perspective-n-Point (PnP)问题，可以将新图像配准到当前模型
    - Triangulation. 三角测量是SfM的关键步骤，因为它通过冗余增加了现有模型的稳定性
    - Bundle Adjustment. BA是摄像机参数Pc和点参数Xk的联合非线性细化，使重投影误差最小化

# Multi-View Stereo(MVS)

> [Pixelwise View Selection for Unstructured Multi-View Stereo (readpaper.com)](https://readpaper.com/pdf-annotate/note?pdfId=709983199334641664&noteId=1991447797942823424)
