---
title: Muti-view Human Body Reconstruction
date: 2023-10-09 16:33:31
tags:
  - ClothedHumans
  - 3DReconstruction
  - PointCloud
categories: HumanBodyReconstruction
---
Terminology
- Human Radiance Fields
- 3D **Clothed Human** Reconstruction | **Digitization**

<!-- more -->

Muti-view input **Idea**

1. **Depth&Normal Estimation**(2K2K)
2. ~~**Implicit Function**(PIFu or NeRF)~~
3. **Generative** : 2D Image --> Encoder --> Decoder --> 3D Mesh

~~深度估计得到多视角深度图，为了得到最终点云，需要进行深度融合，即**需要估计相机位姿**~~
> [touristCheng/DepthFusion: Fuse multiple depth frames into a point cloud (github.com)](https://github.com/touristCheng/DepthFusion)

Muti-view DepthFusion方法(or Implicit Function)需要相机位姿，位姿估计有误差，解决办法：
- ~~BA(Bundle Adjusted)~~
- **点云配准**(Point Cloud Registration)
- Generative approach

# Depth&Normal Estimation

## Human Body Part(2K2K)

将人体分为12个部分，可以只用头、躯干、手臂(4 part)、腿(4 part)和脚(2 part) 五个网络来预测

- **openpose** ... to get keypoint(json) ,then convert json to npy (shape: 31,3)
- 输入原图image，根据pose(.npy)得到仿射变换矩阵(init_affine_2048)，仿射变换矩阵将目标部位移动到相机中心，然后通过centercrop得到part image
    - 原图image下采样后通过网络得到低分辨率法向量图，low normal插值到2k后同样变换得到part low normal
    - part image和part low normal 通过5个part network得到每个部分的part high normal
    - part high normal通过occupy方式得到的权重，求和拼接成high normal

![image.png|222](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231011103025.png)

Note：pose 中脸部只有 2eye、2ear 和 1nose，手部为 4 个 finger

```python
[0, 1, 2, 3, 4] = [nose, L eye, R eye, L ear, R ear]
[5, 6, 7, 8, 9, 10] = [L shoudler, R shoudler, L elbow, R elbow, L wrist, R wrist]
[11, 12, 13, 14, 15, 16] = [L hip, R hip, L knee, R knee, L ankle, R ankle]
[17, 18, 19, 20, 21, 22] = [L big toe, L little toe, L sole, R big toe, R little toe, R sole]
[23, 24, 25, 26, 27, 28, 29, 30] = [L finger 2, 3, 4, 5, R finger 2, 3, 4, 5]
```

根据 pose 将人体分为 12 个部分用5个网络预测:

```python
face[4, 3]
body[6, 5, 12, 11]
arm[5, 7], [7, 9, 23, 24, 25, 26], [6, 8], [8, 10, 27, 28, 29, 30]
leg[11, 13], [13, 15], [12, 14], [14, 16]
foot[17, 18, 19, 15], [20, 21, 22, 16]
```

## Point Cloud Registration

[neka-nat/probreg: Python package for point cloud registration using probabilistic model (Coherent Point Drift, GMMReg, SVR, GMMTree, FilterReg, Bayesian CPD) (github.com)](https://github.com/neka-nat/probreg)
Old Method Library：Probreg is a library that implements point cloud **reg**istration algorithms with **prob**ablistic model.

### SGHR

[WHU-USI3DV/SGHR: [CVPR 2023] Robust Multiview Point Cloud Registration with Reliable Pose Graph Initialization and History Reweighting (github.com)](https://github.com/WHU-USI3DV/SGHR)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231017210035.png)

## PointCloud Surface Reconstruction

[MeshSet — PyMeshLab documentation](https://pymeshlab.readthedocs.io/en/2022.2/classes/meshset.html)
Use pymeshlab to **screened Poisson surface construction**

[nv-tlabs/NKSR: [CVPR 2023 Highlight] Neural Kernel Surface Reconstruction (github.com)](https://github.com/nv-tlabs/NKSR)
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/pictures/20231017200157.png)
