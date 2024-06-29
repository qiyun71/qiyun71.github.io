---
title: Concrete spalling damage detection and seismic performance evaluation for RC shear walls via 3D reconstruction technique and numerical model updating
date: 2024-05-06 20:03:06
tags:
  - 
categories: Other/Mine/Paper
---

| Title     | Concrete spalling damage detection and seismic performance evaluation for RC shear walls via 3D reconstruction technique and numerical model updating                                                                                        |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author    | Weihang Gao; Caiyan Zhang; Xilin Lu; Wensheng Lu                                                                                                                                                                                             |
| Conf/Jour | Automation in Construction                                                                                                                                                                                                                   |
| Year      | 2023                                                                                                                                                                                                                                         |
| Project   | [Concrete spalling damage detection and seismic performance evaluation for RC shear walls via 3D reconstruction technique and numerical model updating - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0926580523004065) |
| Paper     |                                                                                                                                                                                                                                              |

3D reconstruction+Numerical model updating

<!-- more -->

# Conclusion

主要贡献在于，通过信息转移点矩阵的新概念，该方法能够识别混凝土剥落损伤的空间缺陷特征信息，并将缺陷信息转化为已建立的多层壳单元数值模型，实现钢筋混凝土剪力墙的残余承载力评价。据我们所知，这是首次尝试在被检测墙的重建三维点云模型中隐藏的缺陷信息与其相应有限元模型的性能变化之间建立映射关系。
通过实验室条件下直形剪力墙的循环载荷试验，证明了所提方法的有效性。结果表明，基于便携式智能手机和LiDAR扫描仪采集的三维点云数据，该方法能够成功检测被测试件上新出现的混凝土剥落损伤的空间位置和伸展深度。同时，该方法可量化获取新出现的混凝土剥落损伤引起的被检查剪力墙试件的承载力变化。考虑到其巨大的容量，所提方法在钢筋混凝土剪力墙构件的抗震性能评价中具有极好的应用潜力，为钢筋混凝土剪力墙结构的灾后救援和结构维护提供决策依据。


![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240506200043.png)


# Method

通过手机的雷达扫描仪来进行3D Reconstruction，利用收集到的LiDAR点云数据在信息转换点矩阵中标记暴露点，并检测被检测壁上的剥落损坏

## Information transition point matrix


信息过渡点矩阵由被检测墙外表面所包围的几何空间中均匀分布的点构成，可通过设计图纸确定（**就是用均匀间隔将LWH分为mne等份**）

$\begin{cases}\Delta l_x=L\Big/_{m-1}\\\Delta l_y=W\Big/_{n-1}\\\Delta l_z=H\Big/_{e-1}\end{cases}$

信息过渡点矩阵：
![image.png|666](https://raw.githubusercontent.com/qiyun71/Blog_images/main/MyBlogPic/202403/20240506201724.png)


为了实现损伤检测，应通过旋转、平移和缩放操作，将LiDAR扫描仪建立的三维点云模型的坐标系与信息转换点矩阵的坐标系进行匹配：
- R 首先，将点云模型旋转两圈，确保其底部与 xy 平面平行，前端与 yz 平面平行。
- T 然后，将旋转后的点云模型进行平移，确保其下一角能精确定位在原点上，两边能与正象限的坐标轴重合。
- S 最后，进行缩放变换，以减少点云模型与实际组件尺寸之间的误差。

$\mathbf{C}_q=\left(\widehat{\mathbf{C}}_q\mathbf{R}+\mathbf{T}\right)\mathbf{S}$ 对重建出来的点云进行操作

$\left.\left\{\begin{array}{c}\mathbf{R}=\cos(\beta)\mathbf{I}+(1-\cos(\beta))\mathbf{n}\mathbf{n}^T+\sin(\beta)Skew(\mathbf{n})\\\mathbf{T}=[t_x,t_y,t_z]\\\mathbf{S}=Diag(s_x,s_y,s_z)\end{array}\right.\right.$

平移+缩放很好理解，重点是旋转
$\mathbf{n}$: normalized rotation axis column vector $\mathbf{n}=[n_{x},n_{y},n_{z}]^{\mathrm{T}}$
$\mathbf{I}$: 单位矩阵
$Skew(\mathbf{n})=\begin{bmatrix}0&-n_z&n_y\\n_z&0&-n_x\\-n_y&n_x&0\end{bmatrix}$

点云变换结束后，从理论上讲，在未损坏条件下，信息转移点矩阵的所有点都应被点云模型拟合的空间表面所包围。然而，当混凝土剥落损坏发生时，受损区域的表面混凝土缺失，一些混凝土内部成为 LiDAR 扫描仪扫描的表面。因此，对于受损区域，点云将凹入模型中，并暴露信息转换点矩阵中的点



